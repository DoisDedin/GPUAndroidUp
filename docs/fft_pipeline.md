# Pipeline de FFT com pesos dinâmicos

Esta versão do app agora replica a arquitetura do MAD tanto na CPU quanto no TensorFlow Lite para executar FFTs simples com vetores de peso arbitrários. O objetivo é comparar, com dados simulados de **10 sensores enviando 4096 amostras simultaneamente**, o tempo de resposta de cada abordagem e preparar o terreno para integrar isso ao TCC.

## Simulação 10×4096
- `AccelerometerBatchGenerator` (`vulkanfft/src/main/java/com/example/vulkanfft/util/AccelerometerBatchGenerator.kt`) gera os mesmos vetores de acelerômetro usados pelo MAD (timestamps + X/Y/Z) com 4096 amostras por sensor. O `FftInputBuilder` transforma essas janelas em magnitudes float `[10][4096]` e calcula pesos `[10][2049]`. O comprimento 4096 garante potência de 2 para os kernels RFFT do TensorFlow Lite.
- O mesmo lote é reutilizado entre o processamento CPU e TFLite para garantir comparabilidade; basta chamar `FirstViewModel.runFftCpu()` e, na sequência, `runFftTflite(...)`.

## CPU baseline
- `FftCpuProcessor` (`vulkanfft/src/main/java/com/example/vulkanfft/util/FftCpuProcessor.kt`) executa um DFT direto (sem otimizações) para cada sensor, calcula magnitude e aplica o vetor de pesos dinâmico.
- O `FirstViewModel` expõe `runFftCpu()`, que mede a duração, publica o resumo (somatório ponderado de cada sensor e primeiros bins do sensor 0) e serve como baseline para os benchmarks.

## TensorFlow Lite
- O script `app/libs/pythonmodels/make_fft_model.py` gera `vulkanfft/src/main/assets/fft_model.tflite` usando `tf.signal.rfft`. Ele aceita duas entradas `[10,4096]` (sinais) e `[10,2049]` (pesos) e retorna `[10,2049,4]` (real, imag, magnitude e magnitude ponderada). Rode novamente o script sempre que precisar alterar o tamanho do sinal (lembre-se de manter potências de 2).
- `FftTfliteProcessor` (`vulkanfft/src/main/java/com/example/vulkanfft/util/FftTfliteProcessor.kt`) carrega o modelo, aplica o delegate escolhido (CPU/GPU/NNAPI) e reconstrói a mesma estrutura de dados do `FftCpuProcessor` para permitir comparações diretas.
- No app, existem botões para `FFT Lite CPU/GPU/NNAPI`, todos chamando `FirstViewModel.runFftTflite(...)`.

## Observações sobre o MAD
- O pipeline MAD existente funciona, mas o método `getMAD` usa timestamps em **milissegundos** enquanto a janela é calculada em **nanosegundos** (`TimeUnit.SECONDS.toNanos(5)`), o que faz com que todas as amostras caiam em um único bloco. Ajustar para `TimeUnit.SECONDS.toMillis(5)` (ou converter os timestamps para nanos) é essencial para obter blocos de 5 s reais e métricas de desvio padrão corretas.
- Fora isso, os processadores `StatModelProcessor/StatModelReduxProcessor` já encapsulam bem o modelo `mad_model.tflite` e podem servir como referência direta para o novo pipeline de FFT.
