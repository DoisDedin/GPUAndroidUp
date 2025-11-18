    # GPUAndroidUp / VulkanFFT

    Aplicativo Android (API 31+) usado no TCC para comparar diferentes estratégias de processamento estatístico/FFT em dispositivos móveis. Ele expõe pipelines CPU e TensorFlow Lite para duas tarefas principais:

    - **MAD (Mean Absolute Deviation) + estatísticas de magnitude**: versão clássica em Kotlin e versão acelerada via modelo `mad_model.tflite`, executável com delegates CPU/GPU/NNAPI.
    - **FFT com pesos dinâmicos**: baseline CPU (DFT direto) e modelo TFLite (`fft_model.tflite`) que aplica `tf.signal.rfft` em blocos de 10 sensores × 4096 amostras simultâneas (mesmos vetores usados no MAD).

    ## Estrutura rápida

    | Pasta | Descrição |
    |-------|-----------|
    | `app/` | App Android (UI + ViewModels) que aciona benchmarks e mostra os resultados. |
    | `vulkanfft/` | Módulo library com os utilitários Kotlin, assets `.tflite` e código nativo (se necessário). |
    | `app/libs/pythonmodels/` | Scripts Python para gerar/atualizar os modelos TFLite utilizados. |
    | `docs/fft_pipeline.md` | Descreve o pipeline FFT completo e anotações importantes sobre o MAD. |

    ## Requisitos

    1. **Android Studio (Hedgehog/Koala)** com SDK 35 e NDK 26.1 (já referenciado em `vulkanfft/build.gradle.kts`).
    2. **JDK 17** (o projeto usa `compileOptions`/`kotlinOptions` compatíveis com Java 17).
    3. **Python 3.9+** com TensorFlow Lite 2.16 (ou superior) para regenerar os modelos.
    4. **Dispositivo ou emulador** com API 31+ e suporte aos delegates desejados (GPU/NNAPI podem variar por hardware).

    ## Primeira execução

    1. Clone o repositório e abra a pasta raiz no Android Studio.
    2. Aguarde o Sync do Gradle (o wrapper baixa Gradle 8.7 automaticamente).
    3. Conecte um dispositivo ou inicie um emulador com API 31+.
    4. Execute o app (`app` module) normalmente; a tela inicial contém os botões para MAD (CPU, CPU-Optimized, GPU, NNAPI) e FFT (CPU, TFLite CPU/GPU/NNAPI).

    ### Caso o Gradle falhe por permissão

    Alguns ambientes (ex.: macOS com restrições) podem negar escrita em `~/.gradle`. Garanta que o usuário tenha permissão de escrita ou configure `GRADLE_USER_HOME` para uma pasta acessível antes de rodar `./gradlew`.

    ## Gerando/atualizando os modelos TFLite

    Os arquivos são versionados em `vulkanfft/src/main/assets`, mas se quiser regenerar:

    ```bash
    cd app/libs/pythonmodels
    python3 make_mad_model_float.py # cria/atualiza mad_model.tflite (versão float32 compatível com GPU/NNAPI)
    python3 make_fft_model.py      # cria/atualiza fft_model.tflite
    # scripts equivalentes para MAD podem ser adicionados aqui (ver notebooks).
    ```

    Dica: se ocorrerem avisos do Matplotlib/fontconfig, defina `MPLCONFIGDIR=/tmp/mplcache` antes de rodar o script (já suportado no arquivo).

## Suíte de testes exposta no app

A tela principal agrupa os testes em três blocos:

| Grupo | Botões / Cenários | Descrição |
|-------|-------------------|-----------|
| **MAD — Execução individual (10 repetições)** | CPU Kotlin, TFLite CPU, TFLite GPU, TFLite NNAPI | Cada botão executa 10 vezes o MAD com um único pacote de 4096 amostras (1 sensor). O pipeline reutiliza o mesmo lote gerado por `AccelerometerBatchGenerator` para garantir comparabilidade. |
| **MAD — Processar 10 pacotes** | CPU Kotlin x10, TFLite CPU x10, TFLite GPU x10, TFLite NNAPI x10 | Mesmo algoritmo dos botões anteriores, mas cada repetição processa simultaneamente 10 pacotes (10 sensores × 4096 amostras). Ideal para medir amortização do custo de transferência. |
| **FFT — Execução individual / Processar 10 pacotes** | FFT CPU, FFT TFLite CPU/GPU/NNAPI e equivalentes `x10` | O modo individual mede 10 execuções com o lote padrão (10 sensores, 4096 amostras). O modo “x10” cria batches de 10 pacotes para simular envio em lote para o delegate. |

Há ainda o botão **“Executar suíte completa”** que percorre todos os cenários acima em sequência usando as mesmas 10 repetições por cenário. O resultado mais recente aparece no texto “Benchmark” logo abaixo da barra de progresso.

## Métricas coletadas em cada benchmark

A cada execução (manual ou pela suíte) são gravados:

- **Tempo total (média, desvio padrão, min, max)**: tempo agregado por repetição.
- **Tempo de transferência (média, desvio, min, max)**: medido ao copiar amostras/pesos do host para os `ByteBuffer`s de cada delegate.
- **Tempo de processamento (média, desvio, min, max)**: tempo interno da inferência/DFT após a transferência.
- **Throughput**: operações processadas por segundo (considerando 4096 amostras × nº de sensores × tamanho do lote).
- **Iterações e tamanho do lote**: campo `iterations` fica em 10 para os testes normais; `batchSize` vale 1 (execução individual) ou 10 (modo “x10”).
- **Notas**: último resultado do MAD (mean/std/min/max) ou resumo do FFT (somatório dos pesos dos quatro primeiros sensores + bins iniciais).

Todas essas métricas são armazenadas em `benchmark_results.csv` (para consumo em planilhas) e em `benchmark_results.txt` (versão legível). A API usa `BenchmarkReporter` para anexar os metadados de dispositivo (fabricante/modelo, hardware, SoC, SDK, estado do modo economia).

## Teste energético automatizado

O bloco “Teste de gasto energético” possui:

1. **Botão “Iniciar ciclo energético (4 cenários)”**: dispara um job em background que executa, nessa ordem, os cenários MAD CPU Kotlin, MAD TFLite CPU, MAD TFLite GPU e MAD TFLite NNAPI. Cada cenário roda **100 execuções**. Antes e depois de cada bloco são capturados:
   - `% de bateria`, energia em nWh (`BATTERY_PROPERTY_ENERGY_COUNTER`), carga em mAh (`BATTERY_PROPERTY_CHARGE_COUNTER`), temperatura e estado de carregamento.
   - Modo economia de bateria (ligado/desligado) via `PowerManager`.
2. **Botão “Cancelar teste energético”**: cancela imediatamente o job (caso esteja rodando). A UI mostra “Teste energético em execução/parado” e desabilita/habilita os botões automaticamente.
3. **Área de status**: apresenta o texto instrutivo + estado atual, permitindo que o usuário rode uma vez sem modo economia e depois repita com o modo ativado.

Saídas do teste energético:

- `energy_tests.txt`: resumo textual por cenário (100 execuções) com tempo total e queda de bateria.
- `energy_results.csv`: registro estruturado contendo todos os campos coletados (início/fim de bateria, variação, energia, temperatura, se estava carregando, notas, etc.).

Esses arquivos são totalmente separados dos benchmarks normais. O botão “Apagar logs” (via `ResultLogger.clearAll`) limpa ambos os conjuntos, garantindo que o professor possa inspecionar apenas os dados mais recentes.

## Fluxos internos

- **Geração de dados**: `AccelerometerBatchGenerator` cria 10 sensores × 4096 amostras; `FftInputBuilder` converte para magnitudes float e produz pesos dinâmicos.
- **MAD Kotlin**: `getMAD()` calcula magnitude, agrupa em janelas de 5 s (`TimeUnit.SECONDS.toMillis(5)`) e devolve média/desvio/mín/máx.
- **MAD TFLite**: `StatModelProcessor` carrega `mad_model.tflite` e injeta os delegates CPU/GPU/NNAPI; os tempos são obtidos preenchendo `ByteBuffer`s diretos (transferência) e medindo `interpreter.run()` (processamento).
- **FFT CPU**: `FftCpuProcessor.process()` roda o DFT direto e retorna `FftResult`.
- **FFT TFLite**: `FftTfliteProcessor.process()` usa `runForMultipleInputsOutputs` com buffers diretos para mapear real/imag/magnitude/magnitude ponderada.
- **Registro**: `BenchmarkReporter.append()` escreve CSV/TXT das execuções normais; `EnergyReporter.append()` grava as medições energéticas.

Mais detalhes conceituais do pipeline FFT (geração dos pesos, ordem dos tensores, etc.) estão em `docs/fft_pipeline.md`.

## Como operar os testes no app

1. Abra o aplicativo e use os botões dos blocos “MAD” ou “FFT” para rodar testes individuais. Cada clique dispara 10 repetições e atualiza o texto “Benchmark”.
2. Clique em “Executar suíte completa” para percorrer todos os cenários sequencialmente; acompanhe a barra/label de progresso.
3. Para o teste energético, certifique-se de que o celular não está carregando, ajuste o modo economia conforme o cenário desejado e toque em “Iniciar ciclo energético”. O processo leva alguns minutos (400 execuções ao todo). Use “Cancelar” se precisar interromper.
4. Compartilhe os resultados via “Compartilhar logs”; o app junta todos os arquivos CSV/TXT da pasta `benchmarks/`.
5. Caso queira recomeçar, use “Apagar logs” para limpar completamente tanto os benchmarks quanto os registros energéticos.