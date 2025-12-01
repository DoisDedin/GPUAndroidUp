    # GPUAndroidUp / VulkanFFT

    Aplicativo Android (API 31+) usado no TCC para comparar diferentes estratégias de processamento estatístico/FFT em dispositivos móveis. Ele expõe pipelines CPU e TensorFlow Lite para duas tarefas principais:

    - **MAD (Mean Absolute Deviation) + estatísticas de magnitude**: versão clássica em Kotlin e versão acelerada via modelo `mad_model.tflite`, executável com delegates CPU/GPU/NNAPI.
    - **FFT com pesos dinâmicos**: baseline CPU (DFT direto) e modelo TFLite (`fft_model.tflite`) que aplica `tf.signal.rfft` em blocos de 10 sensores × 4096 amostras simultâneas (mesmos vetores usados no MAD).

## Estrutura rápida

| Pasta | Descrição |
|-------|-----------|
| `app/` | App Android (UI + ViewModels) que aciona benchmarks e mostra os resultados. |
| `vulkanfft/` | Biblioteca com utilitários Kotlin, assets `.tflite`, geradores de dados e processadores FFT/MAD. |
| `app/libs/pythonmodels/` | Scripts Python que constroem/atualizam os modelos TFLite (MAD e FFT). |
| `app/src/BANCHMARK/` | Resultados coletados no campo (CSV, TXT, gráficos e scripts de visualização). |
| `scripts/` | Ferramentas auxiliares em Python (ex.: `generate_fft_rfft_models.py`). |
| `tests/` | Artefatos de automação/validação extra. |
| `docs/fft_pipeline.md` | Descrição detalhada do pipeline FFT e notas de projeto. |

## Visão geral do projeto

O objetivo do GPUAndroidUp é medir, de forma reprodutível, como duas tarefas de processamento de sinais (MAD e FFT) se comportam em dispositivos Android reais. A aplicação combina:

- dados sintéticos derivados de acelerômetros (mesmo esquema dos sensores fisiológicos do TCC);
- modelos TFLite especializados para MAD/FFT (gerados em Python e embarcados nos assets);
- um app Android que permite orquestrar cenários (CPU Kotlin, TFLite CPU/GPU/NNAPI, batches x10) e registrar métricas;
- scripts que consolidam os resultados em gráficos comparativos.

## Arquitetura em alto nível

```
[Dados base (AccelerometerBatchGenerator)]
        |
[Scripts Python -> modelos mad_model.tflite / fft_model.tflite]
        |
[Módulo vulkanfft: FftInputBuilder, StatModelProcessor, FftTfliteProcessor]
        |
[App (FirstFragment/FirstViewModel/BenchmarkExecutor)]
        |
[Logs + benchmark_results.csv/.txt -> scripts generate_charts.py]
```

- `vulkanfft` expõe tudo que é comum aos módulos (geração de inputs, FFT CPU, wrappers TFLite).
- `app` injeta esses utilitários, renderiza a tela de testes e fornece os botões/logs.
- `app/src/BANCHMARK` guarda os experimentos e os scripts usados para gerar gráficos/insights.

## Principais classes Kotlin

- `vulkanfft/src/main/java/com/example/vulkanfft/util/AccelerometerBatchGenerator.kt`: gera lotes de 10 sensores × 4096 amostras com ruído controlado, usado tanto por MAD quanto por FFT.
- `vulkanfft/src/main/java/com/example/vulkanfft/util/FftInputBuilder.kt`: converte o lote bruto em tensores float32 (magnitudes, pesos dinâmicos, janelas) compatíveis com os modelos TFLite.
- `vulkanfft/src/main/java/com/example/vulkanfft/util/FftCpuProcessor.kt`: implementação de referência (DFT) usada para validar resultados e como fallback.
- `vulkanfft/src/main/java/com/example/vulkanfft/util/StatModelProcessor.kt`: carrega `mad_model.tflite` e provê inferências com delegates CPU/GPU/NNAPI.
- `vulkanfft/src/main/java/com/example/vulkanfft/util/FftTfliteProcessor.kt`: wrapper sobre o interpreter FFT, cuidando de buffers diretos e contabilidade de transferência/compute.
- `app/src/main/java/com/example/vulkanfft/BenchmarkExecutor.kt`: orquestra cada cenário, mede tempos, calcula estatísticas e salva logs.
- `app/src/main/java/com/example/vulkanfft/FirstViewModel.kt`: guarda as preferências de iteração/lote, expõe progresso e chama o executor.
- `app/src/main/java/com/example/vulkanfft/FirstFragment.kt`: tela com todos os botões (MAD, FFT, suíte completa, testes energéticos, compartilhamento/limpeza de logs).
- `app/src/main/res/layout/fragment_first.xml`: layout com os grupos de chips, botões, indicadores e seções de log que aparecem na tela inicial.

## Pipeline de dados (acelerômetro ➜ entrada do modelo)

1. `AccelerometerBatchGenerator` cria pacotes determinísticos de acelerômetro (10 sensores × N amostras) replicando padrões de sinais fisiológicos.
2. `BenchmarkExecutor` decide se o cenário é single ou x10 e chama:
   - `buildMadInputs`, que converte o lote em leituras discretas para `getMAD` (CPU) ou tensores `[timestamps, x, y, z]` para o `StatModelProcessor`;
   - `buildFftInputs`, que usa `FftInputBuilder` para montar os tensores (real, imag, magnitude ponderada).
3. Durante a execução:
   - CPU Kotlin roda `getMAD`/`FftCpuProcessor.process()` e mede apenas `compute_ms`;
   - TFLite mede transferência e processamento separadamente via `StatModelProcessor`/`FftTfliteProcessor` e salva cada amostra em `InferenceTiming`.
4. O executor consolida as estatísticas (`TimingStats`) e chama `BenchmarkReporter`/`ResultLogger` para registrar CSV/TXT e logs human-readable.

Os vetores usados nos experimentos são derivados de acelerômetro, mas o mesmo pipeline funciona para PPG/ECG bastando adaptar o gerador de entradas (há pontos de extensão documentados em `docs/fft_pipeline.md`).

## Modelos TFLite e scripts Python

- Scripts principais: `app/libs/pythonmodels/make_mad_model_float.py`, `app/libs/pythonmodels/make_fft_model.py` e `scripts/generate_fft_rfft_models.py`.
- Requisitos: Python 3.9+, TensorFlow 2.16.1, NumPy, SciPy, matplotlib, seaborn.
- `make_mad_model_float.py` lê amostras de acelerômetro, constrói um modelo TFLite com a mesma lógica do MAD Kotlin (sem quantização para manter compatibilidade com GPU/NNAPI) e exporta para `vulkanfft/src/main/assets/mad_model.tflite`.
- `make_fft_model.py` usa `tf.signal.rfft` para calcular FFT de 10 sensores simultâneos, adiciona camadas de normalização/aplicação de pesos e exporta `fft_model.tflite` com as assinaturas esperadas pelo app.
- `scripts/generate_fft_rfft_models.py` serve como laboratório para recalibrar pesos, testar quantização e gerar múltiplas variantes (útil quando se quer comparar FP32 vs FP16).
- Sempre que atualizar os modelos, rode `./gradlew :vulkanfft:assemble` para garantir que os assets sejam empacotados corretamente.

## Interface e automação dos testes

- **Chips superiores**: controlam `iterations` e `batchSize`. Qualquer clique nos botões usa exatamente esses valores, e a suíte completa percorre 16 cenários × 3 escalas com as mesmas configurações.
- **Botões MAD/FFT**: chamam `BenchmarkExecutor.runScenario` com o `DataScale` selecionado (1x, 2x ou 4x) e salvam os logs em `app/src/main/assets/benchmarks`.
- **Botão “Executar suíte completa”**: percorre todos os cenários e alimenta a barra de progresso (`FirstViewModel.BenchmarkProgress`). Resultado recente aparece em `textBenchmarkResult`.
- **Teste energético**: `EnergyTestService` executa quatro cenários MAD (CPU, TFLite CPU/GPU/NNAPI) com 100 execuções cada, capturando bateria, energia, temperatura e modo de economia. Os logs vão para `energy_tests.txt` e `energy_results.csv`.
- **Compartilhar/Apagar logs**: `ResultLogger` gerencia os arquivos em `Android/data/.../files/benchmarks/`, permitindo exportar ou limpar tudo direto da UI.

## Scripts de análise e reprodutibilidade

- Cada pasta em `app/src/BANCHMARK/<device>-<data>/` contém o CSV bruto + `generate_benchmarks_charts.py` (versão standalone que gera gráficos específicos do dispositivo).
- `generate_charts.py` na raiz aceita qualquer CSV consolidado (como `comparativo-30-11/benchmark_results.csv`) e produz:
  - gráficos por dispositivo (`global`, `tflite`, `speedup`);
  - comparativos (`summary/tempo_*` e `summary/speedup_*`);
  - relatórios textuais com as tabelas consolidadas por vetor/tamanho.
- `generate_overview_charts.py` cria gráficos “densos” multi-dispositivo, comparando simultaneamente Galaxy S21, Moto G04s e Moto G84 para cada algoritmo/delegate (versões single e batch). Esses PNGs ficam em `docs/charts/comparativo-30-11/overview/<ALG>/<DELEGATE>/`.
- Para reproduzir os gráficos da campanha de 30/11:

```bash
python3 generate_charts.py \
  --csv app/src/BANCHMARK/comparativo-30-11/benchmark_results.csv \
  --output app/src/BANCHMARK/comparativo-30-11/charts

python3 generate_overview_charts.py \
  --csv app/src/BANCHMARK/comparativo-30-11/benchmark_results.csv \
  --output docs/charts/comparativo-30-11/overview
```

Os scripts configuram `MPLCONFIGDIR` localmente para não depender de `~/.matplotlib`, permitindo rodar em ambientes restritos. Para facilitar a navegação, uma cópia dos gráficos comparativos entre os três dispositivos está em `docs/charts/comparativo-30-11/summary/` (mesmos arquivos `tempo_*` e `speedup_*` produzidos pelo script), prontos para referência no TCC.

## Dependências e toolchain

| Componente | Versão |
|-----------|--------|
| Gradle Plugin Android | 8.6.0 |
| Kotlin | 1.9.0 |
| Compile/Target SDK | 35 / 34 (minSdk 31) |
| Java / Kotlin JVM | 17 |
| Jetpack | Core KTX 1.16.0, AppCompat 1.7.0, Material 1.12.0, ConstraintLayout 2.2.1, Navigation 2.8.9 |
| Testes | JUnit 4.13.2, AndroidX Test 1.2.1, Espresso 3.6.1 |
| TensorFlow Lite | 2.16.1 (`tensorflow-lite`, `-gpu`, `-gpu-api`) |
| FFT CPU extra | `com.github.wendykierp:JTransforms:3.1` |

Ferramentas externas: Android Studio Hedgehog/Koala, SDK 35, NDK 26.1 (quando necessário). Para Python, recomenda-se criar um ambiente virtual (`python3 -m venv .venv && source .venv/bin/activate`) e instalar `tensorflow==2.16.1 numpy scipy matplotlib seaborn pandas`.

## Visão prática para novos colaboradores

- Clone o repositório, abra na IDE e execute o módulo `app`.
- Use os chips para controlar iterações/lote e clique nos botões desejados; os logs aparecem na própria tela e são salvos automaticamente.
- Rode `generate_charts.py` ou os scripts específicos de cada pasta para transformar os CSVs em gráficos (PNG) e tabelas para relatórios.
- Para atualizar os modelos, entre em `app/libs/pythonmodels`, execute os scripts Python e depois rode `./gradlew :vulkanfft:assemble` para empacotar os novos assets.
- Inspecione `app/src/BANCHMARK/` para exemplos de execuções reais (cada pasta contém CSV, TXT, gráficos e o script usado). Isso serve como referência de formato para novas campanhas.


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

A tela principal agrupa os testes em três blocos. No topo da tela existem dois grupos de chips (iteração e tamanho de lote) com as opções 1/4/8/12. O valor escolhido para **iterações** define quantas vezes cada cenário é repetido sempre que um botão é pressionado ou quando a suíte completa é executada. O valor escolhido para **lote** só é aplicado aos cenários “x10”: cada repetição processa sucessivamente o número selecionado de pacotes (por exemplo, 12 pacotes = 12 × 10 sensores × 4096/8192/16384 pontos). Os cenários “single” continuam processando apenas 1 pacote, porém respeitam o número de iterações configurado.
| Grupo | Botões / Cenários | Descrição |
|-------|-------------------|-----------|
| **MAD — Execução individual** | CPU Kotlin, TFLite CPU, TFLite GPU, TFLite NNAPI | Executa MAD em um único pacote (1 sensor × 4096/8192/16384 amostras) e repete conforme o chip de iterações escolhido. |
| **MAD — Processar N pacotes** | CPU Kotlin x10, TFLite CPU x10, TFLite GPU x10, TFLite NNAPI x10 | Mesmo algoritmo, porém cada repetição percorre N pacotes sequenciais (N definido no chip superior). Útil para medir amortização de transferência. |
| **FFT — Execução individual / Processar N pacotes** | FFT CPU, FFT TFLite CPU/GPU/NNAPI e equivalentes `x10` | FFT sempre usa 10 sensores × 4096/8192/16384 amostras e repete o cenário conforme o chip de iterações. Nos botões x10 o app processa N pacotes consecutivos usando o delegate selecionado. |

Há ainda o botão **“Executar suíte completa”** que percorre todos os 16 cenários acima combinados com os 3 tamanhos de vetor (1x/2x/4x). Assim, com os chips ajustados para 12 iterações e 12 pacotes, o app executa 48 cenários diferentes e cada um deles roda 12 vezes. O resultado mais recente aparece no texto “Benchmark” logo abaixo da barra de progresso, acompanhado da barra/label de progresso.

## Métricas coletadas em cada benchmark

A cada execução (manual ou pela suíte) são gravados:

- **Tempo total (média, desvio padrão, min, max)**: tempo agregado por repetição.
- **Tempo de transferência (média, desvio, min, max)**: medido ao copiar amostras/pesos do host para os `ByteBuffer`s de cada delegate.
- **Tempo de processamento (média, desvio, min, max)**: tempo interno da inferência/DFT após a transferência.
- **Throughput**: operações processadas por segundo (considerando 4096 amostras × nº de sensores × tamanho do lote).
- **Iterações e tamanho do lote**: `iterations` recebe exatamente o valor escolhido no chip superior (1/4/8/12). `batchSize` vale 1 para os cenários individuais e passa a refletir o número de pacotes selecionado quando o cenário é “x10” (ex.: 12 pacotes ⇒ `batchSize=12`).
- **Notas**: último resultado do MAD (mean/std/min/max) ou resumo do FFT (somatório dos pesos dos quatro primeiros sensores + bins iniciais).

Todas essas métricas são armazenadas em `benchmark_results.csv` (para consumo em planilhas) e em `benchmark_results.txt` (versão legível). A API usa `BenchmarkReporter` para anexar os metadados de dispositivo (fabricante/modelo, hardware, SoC, SDK, estado do modo economia).

## Resultados dos benchmarks de 30-11

Os arquivos em `app/src/BANCHMARK/comparativo-30-11/` consolidam a campanha rodando em 30/11 nos dispositivos Galaxy S21 (Exynos 2100), Moto G04s (Spreadtrum T606) e Moto G84 5G (Snapdragon 695/SM6375). Para esta rodada:

- Os chips superiores foram configurados com **12 iterações** e **12 pacotes**.
- O botão “Executar suíte completa” percorreu todos os 16 cenários combinados com as escalas 1x, 2x e 4x, resultando em 48 passos × 12 repetições por dispositivo.
- Nos cenários `x10`, cada repetição processou 12 pacotes consecutivos; isso aparece como `batch_size=12` e nas descrições (`12×(...)`) dentro do CSV.

### Galaxy S21 (30-11)

- **MAD 1× (4096 pts, batch=1)**: TFLite CPU e GPU ficaram em ~0,64 ms enquanto o CPU Kotlin levou 2,70 ms e o NNAPI 2,53 ms (`iterations=12` em todos os casos). Ao escalar para 16 384 pontos (4×) o TFLite GPU manteve 2,56 ms contra 24,56 ms do CPU puro.
- **MAD 1× x10 (12 pacotes)**: TFLite CPU/GPU ≈7,6 ms por repetição versus 25,5 ms no CPU Kotlin e 45,2 ms no NNAPI, evidenciando o ganho de amortização no delegate.
- **FFT 1× (10 sensores × 4096 pts)**: todos os delegates TFLite ficaram entre 2,46 ms e 2,46 ms, metade do CPU Kotlin (4,09 ms). Com batches de 12 pacotes os tempos cresceram para ~31–33 ms, porém os delegates mantiveram leve vantagem sobre a CPU nativa.

### Moto G04s (30-11)

- **MAD 1× single**: CPU Kotlin demorou 121,5 ms, enquanto TFLite CPU/GPU ficaram em ~60 ms e o NNAPI em 62 ms. Nos tamanhos maiores os delegates escalam melhor (por exemplo, 4×: GPU 8,2 ms vs CPU 85,7 ms).
- **MAD 1× x10**: o GPU entrega 28,3 ms, superando CPU (192 ms) e NNAPI (226 ms). O TFLite CPU apresentou 120 ms neste modo, mostrando forte impacto do custo de transferência nesse hardware.
- **FFT**: no tamanho base, GPU = 8,28 ms, CPU Kotlin = 82,96 ms e NNAPI = 13,58 ms. Com batches de 12 pacotes os delegates variam entre 97–163 ms, ainda assim muito abaixo do CPU (230 ms), reforçando que o SoC é limitado por CPU escalar.

### Moto G84 5G (30-11)

- **MAD 1× single**: GPU/CPU/NNAPI TFLite ficaram praticamente empatados perto de 1,0 ms contra 4,12 ms do CPU Kotlin. Em 4×, os delegates ficaram em ~3,9 ms enquanto a CPU subiu para 35,7 ms.
- **MAD 1× x10**: cerca de 12 ms em qualquer delegate TFLite, frente a 49 ms no CPU Kotlin.
- **FFT**: o cenário base registrou 3,88–3,91 ms para os delegates e 7,95 ms na CPU; com 12 pacotes os delegates ficaram em ~46,7 ms versus 86,3 ms no CPU Kotlin.

Todos os gráficos (.png) gerados automaticamente estão em `app/src/BANCHMARK/comparativo-30-11/charts/` (por dispositivo e na pasta `summary/`), mostrando as curvas completas de tempo, transferência e speedup usados para produzir os números acima.

### Principais insights experimentais

- **Transferência domina o custo**: em todos os dispositivos, 70–85 % do tempo dos delegates TFLite é gasto transferindo os buffers do acelerômetro (ver `tflite_transfer_*.png`). Isso explica por que GPU e NNAPI nem sempre superam o TFLite CPU – o gargalo não é o cálculo da FFT, mas o I/O.
- **Batching amortiza o gargalo**: gráficos como `summary/tempo_FFT_TFLite GPU_49152.png` mostram que processar 12 pacotes sequenciais mantém o tempo por pacote praticamente constante, enquanto o CPU Kotlin cresce linearmente. Estratégias de batching passam a ser o principal “ganho” quando o delegate já é limitado por transferência.
- **Dispositivos low-cost lucram mais**: `summary/speedup_FFT_TFLite GPU.png` evidencia que o Moto G04s atinge >10× de speedup contra CPU, enquanto o Galaxy S21 fica próximo de 1,6×. Ou seja, a escolha do delegate deve considerar o perfil de hardware e não apenas “usar GPU porque é melhor”.
- **Estabilidade estatística**: os desvios padrão (<0,2 ms para delegates) e as 12 repetições configuradas nos chips superiores garantem dados reprodutíveis. Isso é importante para serviços contínuos (JourneyService) e para evidenciar que os resultados não são outliers.
- **Base em acelerômetro, aplicável a outros sensores**: todos os pacotes usados vêm do `AccelerometerBatchGenerator` (10 sensores × 4096 amostras), mas a mesma metodologia se aplica a PPG, ECG ou quaisquer fluxos de séries temporais, bastando ajustar o gerador de entradas.

Essas conclusões sustentam o texto do TCC: o ganho real vem de amortizar transferência e escolher delegates compatíveis com o hardware-alvo. Em aplicações médicas, wearables ou safety (detecção de quedas, ruídos industriais), isso se traduz em menor latência e menor consumo energético no edge, permitindo executar análises FFT em tempo real sem depender da nuvem.

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

1. Abra o aplicativo e use os botões dos blocos “MAD” ou “FFT” para rodar testes individuais. Antes de começar, escolha no topo quantas iterações (1/4/8/12) e, se desejar testar as versões `x10`, quantos pacotes consecutivos cada repetição deve processar. Cada clique usa exatamente essas configurações e atualiza o texto “Benchmark”.
2. Clique em “Executar suíte completa” para percorrer todos os cenários sequencialmente; acompanhe a barra/label de progresso e lembre-se de que a suíte executa 16 cenários × 3 tamanhos (48 passos) respeitando as opções de iteração/lote selecionadas.
3. Para o teste energético, certifique-se de que o celular não está carregando, ajuste o modo economia conforme o cenário desejado e toque em “Iniciar ciclo energético”. O processo leva alguns minutos (400 execuções ao todo). Use “Cancelar” se precisar interromper.
4. Compartilhe os resultados via “Compartilhar logs”; o app junta todos os arquivos CSV/TXT da pasta `benchmarks/`.
5. Caso queira recomeçar, use “Apagar logs” para limpar completamente tanto os benchmarks quanto os registros energéticos.

## Testes automatizados e tolerâncias

### Unit tests (Gradle)

Rode `./gradlew :app:testDebugUnitTest` para validar:

| Arquivo | O que valida |
|---------|--------------|
| `FftValidationUnitTest` | FFT CPU vs DFT manual, aplicação de pesos e todos os modos de normalização (`NONE`, `/N`, `/√N`). |
| `FftInputBuilderTest` | Conversão de sensores para magnitudes/weights e validação de erros quando faltam amostras. |
| `MadValidationUnitTest` | `BenchmarkExecutor.getMAD` contra uma implementação manual com janelas de 5 s e determinismo por semente. |

Todos esses testes rodam em JVM pura; logs do `FftCpuProcessor` caem no console (fallback de `println`) para evitar dependência de `android.util.Log`.

### Instrumented tests (device)

`./gradlew :app:connectedAndroidTest` executa `TfliteDelegatesInstrumentedTest`, que compara FFT CPU × FFT TFLite com métricas detalhadas:

- **Métricas logadas**: `maxRelativeDiff`, `meanRelativeDiff`, `maxAbsoluteDiff`, `meanAbsoluteDiff` e `rmse`, além dos oito primeiros bins e suas razões.
- **Tolerâncias**:
  - CPU vs CPU: 1,5 %
  - GPU: 4 % (aceita FP16/FPmix)
  - NNAPI: 3 %

Os logs aparecem na tag `FFT_TEST`; use qualquer leitor de logcat ou `adb logcat | grep FFT_TEST` para capturá-los e anexar aos relatórios.

#### Reduzindo o tempo dos testes

As execuções FFT CPU podem ser demoradas em dispositivos mais lentos. Por padrão, os testes instrumentados usam 4 sensores e apenas o tamanho 4096. Para rodar a suíte completa (10 sensores e tamanhos 4096/8192/16384), passe o argumento `fftExtended=true`:

```bash
# Via Gradle
./gradlew :app:connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.fftExtended=true

# Ou direto via adb
adb shell am instrument -w \
  -e fftExtended true \
  com.example.vulkanfft.test/androidx.test.runner.AndroidJUnitRunner
```

A primeira execução do teste imprime a configuração escolhida em `FFT_TEST`. Documente nos relatórios qual modo foi utilizado.
