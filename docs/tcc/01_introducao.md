1 Introdução

1.1 Contextualização e motivação

A popularização de dispositivos móveis e vestíveis habilita cenários de computação em borda nos quais a coleta, o processamento e a interpretação de sinais ocorrem localmente, sem dependência constante de infraestrutura em nuvem. Aplicações de monitoramento fisiológico, ergonomia industrial, segurança do trabalho e telemetria pessoal demandam respostas em tempo real e preservação de privacidade, o que torna o processamento no próprio aparelho um diferencial. Nesse contexto, operações clássicas como Mean Absolute Deviation (MAD) e Fast Fourier Transform (FFT) permanecem essenciais para extrair indicadores de variabilidade e de conteúdo espectral a partir de séries multissensores, possibilitando alertas imediatos sobre padrões anômalos sem transmitir dados sensíveis. Executar esse tipo de análise diretamente em smartphones e wearables também contribui para reduzir latências, economizar energia de comunicação e garantir continuidade de serviço quando a conectividade é limitada.

1.2 Problema de pesquisa

Embora a literatura apresente inúmeros estudos sobre inferência com redes neurais em dispositivos móveis, existem lacunas na avaliação detalhada de blocos fundamentais de processamento de sinais em ambientes Android reais. O problema central deste trabalho consiste em analisar empiricamente como diferentes estratégias de execução — CPU Kotlin versus delegates TensorFlow Lite para CPU, GPU e NNAPI, com e sem processamento em lote — comportam-se ao implementar MAD e FFT em sinais multissensores, considerando tempos de execução e implicações energéticas. A investigação concentra-se em entender o impacto do hardware, do tamanho dos vetores e do uso de batching na eficiência global dos pipelines.

1.3 Objetivos

O objetivo geral é avaliar o desempenho e o comportamento de pipelines de MAD e FFT em dispositivos Android, combinando implementações nativas em Kotlin com delegates do TensorFlow Lite, de modo a observar tempos de execução, efeitos de lotes e estimativas de consumo energético. Especificamente, busca-se comparar o desempenho dos delegates CPU, GPU e NNAPI frente ao baseline em CPU Kotlin; examinar como as variações de tamanho de vetor (4096, 8192 e 16384 amostras) influenciam o tempo de processamento e a estabilidade das estatísticas; investigar o efeito do processamento em lote (modos x10) como estratégia de amortização de transferência de dados; e realizar uma análise exploratória do impacto energético ao executar os mesmos cenários em diferentes perfis de hardware.

1.4 Justificativa

A relevância deste estudo decorre da necessidade crescente de soluções confiáveis para monitoramento contínuo em saúde, ergonomia e segurança, nas quais decisões rápidas e localmente embasadas podem prevenir eventos críticos. Ao focar em operações básicas como MAD e FFT — frequentemente tratadas como módulos auxiliares nas aplicações — o trabalho fornece uma visão granular do custo computacional real em smartphones e aponta caminhos para arquiteturas híbridas que alternam entre CPU e aceleradores. Além disso, os resultados contribuem para orientar a escolha de delegates adequados a diferentes faixas de hardware, oferecendo subsídios práticos para equipes que precisam equilibrar latência, consumo de energia e complexidade de desenvolvimento em dispositivos de entrada, intermediários e topo de linha.

1.5 Estrutura do trabalho

O Capítulo 2 apresenta a revisão bibliográfica sobre computação em borda, processamento de sinais fisiológicos, uso de MAD/FFT e suporte do TensorFlow Lite em dispositivos móveis. O Capítulo 3 descreve a metodologia adotada, incluindo ambiente de desenvolvimento, arquitetura do sistema, modelos avaliados e procedimentos experimentais. O Capítulo 4 discute os resultados obtidos nos benchmarks e nos testes energéticos, relacionando-os aos diferentes perfis de hardware. Por fim, o Capítulo 5 sintetiza as conclusões, aponta limitações e propõe linhas de trabalho futuras.
