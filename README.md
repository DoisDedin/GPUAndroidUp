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
python3 make_fft_model.py      # cria/atualiza fft_model.tflite
# scripts equivalentes para MAD podem ser adicionados aqui (ver notebooks).
```

Dica: se ocorrerem avisos do Matplotlib/fontconfig, defina `MPLCONFIGDIR=/tmp/mplcache` antes de rodar o script (já suportado no arquivo).

## Fluxos principais

### MAD
- `FirstViewModel.runCPU()` → executa o MAD em Kotlin puro (verifique `getMAD`: atualmente as janelas usam timestamps em ms vs. janela em nanos, ajustar para `TimeUnit.SECONDS.toMillis(5)` torna os blocos corretos).
- `FirstViewModel.runBenchmark()` → instancia `StatModelProcessor` com o delegate escolhido e roda o modelo `mad_model.tflite` repetidas vezes para calcular média/desvio de latência.

### FFT com pesos dinâmicos
- Gerador de dados: `AccelerometerBatchGenerator.generate()` (10 sensores × 4096 amostras). O `FftInputBuilder` converte cada sensor em magnitudes float e gera pesos dinâmicos baseados na energia da janela.
- CPU baseline: `FftCpuProcessor.process()` (DFT direto com pesos).
- TensorFlow Lite: `FftTfliteProcessor.process()` carrega `fft_model.tflite` e aplica GPU/NNAPI/CPU+XNNPACK.
- Interface: `FirstFragment` expõe botões para cada delegate e mostra no log/result os tempos médios + amostras dos espectros ponderados.

Mais detalhes estão em `docs/fft_pipeline.md`.

## Execução de benchmarks

1. Abra o app e clique em “Rodar CPU” para medir a versão Kotlin do MAD.
2. Use “Rodar na CPU-OPTIMIZE / GPU / NumPy (NNAPI)” para comparar os delegates no modelo MAD.
3. Use “FFT CPU (10×4096)” para o baseline e “FFT Lite CPU/GPU/NNAPI” para a versão TFLite. Todos reutilizam o mesmo lote de sensores (os mesmos vetores do MAD) para comparabilidade.
4. Inspecione os logs (`Logcat` tag `MAD`, `BenchmarkProcessor2`, `FftTfliteProcessor`) para ver os resultados completos e métricas de duração.

## Solução de problemas

- **Erro `IsPowerOfTwo(fft_length_data[1])`:** o kernel `RFFT` do TensorFlow Lite exige que o comprimento da FFT seja potência de 2. O projeto usa 4096, mas qualquer potência de 2 funciona desde que atualize `make_fft_model.py` e `FirstViewModel.FFT_SIGNAL_LENGTH`.
- **`GpuDelegateFactory$Options` não encontrado:** garanta que o módulo `vulkanfft` dependa tanto de `org.tensorflow:tensorflow-lite-gpu` quanto de `org.tensorflow:tensorflow-lite-gpu-api` (já configurado no `build.gradle.kts`) e sincronize o projeto.

## Próximos passos sugeridos

- Ajustar a conversão de timestamps no `getMAD` para evitar que todas as amostras caiam em um único bloco.
- Se quiser outros tamanhos de FFT, altere `NUM_SENSORS` e `SIGNAL_LENGTH` em `make_fft_model.py`, gere novamente o modelo e atualize as constantes em `FirstViewModel`/`FftCpuProcessor`.
- Adicionar testes instrumentados que validem o comparativo CPU vs. TFLite para prevenir regressões (ex.: comparar somatório de magnitudes ponderadas).
