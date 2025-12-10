4 Resultados e Discussão

4.1 Configuração dos benchmarks

A campanha oficial de 09/12 percorreu **128 cenários por dispositivo** (16 botões × oito escalas), com os chips superiores fixados em 12 iterações por cenário e lotes de 12 pacotes para os modos “x10”. Os testes foram realizados nos três aparelhos físicos descritos na Metodologia (Galaxy S21, Moto G04s e Moto G84 5G). Cada cenário contemplou comprimentos de 512, 1 024, 2 048, 4 096, 8 192, 16 384, 32 768 e 65 536 pontos por sensor, mantendo 10 sensores simultâneos para FFT. Como todos os dispositivos saturaram a RAM ao ultrapassar 65 k, as escalas de 128 k, 256 k e 524 k ficaram restritas a experimentos manuais e não compõem a suíte oficial. Para reduzir variância, cada botão foi acionado cinco vezes consecutivas (5 × 128 execuções), o que resultou em 640 linhas por dispositivo (816 no Moto G84, por conter também as tentativas extremas documentadas no Apêndice). Todas as execuções single processam um único pacote por iteração, enquanto as variantes x10 reaproveitam os mesmos parâmetros para 12 pacotes consecutivos, preservando o determinismo do gerador e permitindo avaliar amortização de transferência. Os CSVs e gráficos resultantes evidenciam os mesmos padrões observados nas seções anteriores: crescimento linear no baseline em CPU e platôs nos delegates TFLite quando a granularidade é grossa, conforme o modelo conceitual da Seção 2.7.

4.2 Resultados de MAD

4.2.1 Galaxy S21

No Galaxy S21, a coleta de 09/12 mostrou que os delegates mantêm vantagem consistente em toda a faixa 512 → 65 k. Em 4 096 pontos single, a CPU Kotlin levou 2,7 ms, enquanto TFLite CPU/GPU rodaram em ~0,63 ms (speedup ≈4×); em 65 536 pontos single, o delegate GPU permaneceu em **14,9 ms** contra 1652 ms na CPU x10, o que reforça a eficiência da granularidade grossa. Nos modos x10, os tempos passaram de 25–30 ms (CPU) para menos de 8 ms nos delegates nas escalas pequenas, e de 1,65 s para ~183 ms na escala 65 k. As séries de desvio padrão permaneceram abaixo de 1 %, confirmando a estabilidade obtida com 12 repetições × 5 cliques. Esses resultados mostram que, embora o Exynos 2100 possua CPU competitiva, o delegate GPU ainda reduz o tempo ativo quando a janela cresce, exatamente como previsto pelo modelo de amortização da Seção 2.7.

4.2.2 Moto G04s

No Moto G04s, os números reforçam o quanto a GPU compensa a limitação do CPU quad-core. Em 4 096 pontos single a CPU consumiu 121 ms, enquanto a GPU ficou ao redor de 60 ms e o delegate NNAPI em 62 ms; ao subir para 65 536 pontos single a CPU ultrapassou 3,89 s, enquanto o delegate GPU concluiu a mesma carga em **33,5 ms** (speedup >100×). Nos modos x10, o CPU ultrapassou 200 ms em 4 k e 3,9 s em 65 k, enquanto GPU e TFLite CPU ficaram entre 28–400 ms, confirmando a importância do batching para amortizar transferências. As métricas energéticas também apontaram queda de consumo quando a GPU assume o MAD, alinhando desempenho e eficiência no dispositivo de entrada.

4.2.3 Moto G84 5G

O Moto G84 5G exibiu comportamento homogêneo entre delegates. Em 4 096 pontos single, a CPU Kotlin consumiu 4,1 ms e todos os delegates ficaram próximos de 1 ms (≈4×). Em 65 536 pontos single, a CPU atingiu 2,07 s, enquanto os delegates mantiveram ~46 ms (x10) e ~16 ms (single). Como os três delegates compartilham o mesmo gargalo de transferência, as diferenças ficaram abaixo de 2 %, e o ganho sobre a CPU derivou da redução do tempo de computação. As tentativas acima de 65 k falharam por memória, reforçando que o Moto G84 opera confortavelmente até esse limite, mas demanda pipelines específicos para escalas maiores.

4.3 Resultados de FFT

Como discutido na Metodologia, não há kernel FFT dedicado para GPU/NNAPI na versão atual do TensorFlow Lite; assim, todos os delegates fazem fallback para o backend XNNPACK em CPU. Isso significa que as diferenças de tempo decorrem majoritariamente do custo de transferência e das particularidades de cada driver (FP16, FP32, sincronização com GPU). Entre 70 % e 85 % do tempo total continua concentrado em copiar buffers, o que aproxima os resultados dos três delegates e limita os ganhos quando não há batching. Ainda assim, cada dispositivo apresenta nuances próprias.

4.3.1 Galaxy S21

No Galaxy S21, o baseline FFT em Kotlin atingiu 4–5 ms com 4 096 pontos single e ~13 ms em 16 384 pontos. Os delegates CPU/GPU/NNAPI ficaram na faixa de 2,4–2,6 ms em 4 k e 10–11 ms em 16 k, mantendo speedups modestos (≈1,6×) devido ao gargalo de I/O. Nas escalas mais altas (32 k e 65 k) os tempos convergiram para ~100 ms (single) e ~1,18 s (x10), enquanto a CPU saltou para 160 ms e 1,65 s, respectivamente. Esses números mostram que o Exynos 2100 só colhe ganhos relevantes quando amortiza bastante a transferência — cenário típico ao processar 65 k pontos ou lotes x10.

4.3.2 Moto G04s

O Moto G04s voltou a ser o que mais se beneficia do fallback TFLite, pois sua CPU ARM limita a FFT pura. Em 4 096 pontos single, o baseline levou 83 ms contra ~8 ms nos delegates (≈10×). Em 65 536 pontos single, a CPU ultrapassou 2,49 s enquanto os delegates permaneceram próximo de 250 ms. Nos modos x10, o CPU precisou de 230 ms (4 k) e 2,97 s (65 k), ao passo que os delegates variaram entre 97 ms e 2,97 s, com vantagem clara quando a janela cresce. Apesar de todo o processamento ocorrer no CPU do aparelho, a forma como cada delegate organiza buffers reduz o tempo ativo do aplicativo e entrega speedups práticos no dispositivo de entrada.

4.3.3 Moto G84 5G

No Moto G84, os três delegates TFLite ficaram praticamente empatados porque o Snapdragon 695 já oferece FP32 eficiente e o custo dominante é a transferência. Em 4 096 pontos single, a CPU precisou de 7,9 ms e os delegates de 3,9 ms; em 65 536 pontos single, 2,07 s contra ~100 ms. O modo x10 seguiu a mesma tendência (86 ms/644 ms na CPU versus 46 ms/197 ms nos delegates). Assim, mesmo que tudo rode no CPU, o uso do interpreter TFLite reduz a sobrecarga de cópias internas e garante speedups próximos de 2× em todo o espectro avaliado.

4.4 Resultados energéticos

Nesta campanha não executamos o ciclo dedicado de 100 execuções do “Teste energético”; as etiquetas energéticas foram derivadas das leituras que o aplicativo registra junto a cada benchmark (12 execuções por cenário), incluindo `% bateria`, `energy_counter`, `charge_counter` e as temperaturas inicial/final. Esses registros alimentam diretamente as classificações presentes nos CSVs e gráficos: CPU Kotlin rotulada como “Alta”, TFLite CPU como “Média”, TFLite GPU como “Baixa” e NNAPI como “Baixa/Média”. Mesmo com essa coleta enxuta, as leituras evidenciaram que delegar reduz o tempo ativo da CPU, o que resulta em menores quedas de bateria e temperaturas mais estáveis em sessões prolongadas. Tais efeitos reforçam que granularidade grossa, ao amortizar o custo fixo de transferência, contribui diretamente para reduzir o tempo ativo dos componentes e, por consequência, o consumo energético.

4.5 Discussão geral

Os resultados demonstram forte dependência do hardware-alvo. O Moto G04s, limitado por CPU, é o que mais se beneficia de delegar para GPU ou mesmo para o backend TFLite CPU, alcançando speedups superiores a 10× em MAD e FFT. O Galaxy S21 e o Moto G84 5G apresentam CPUs mais capazes e, por consequência, ganhos moderados em FFT quando a transferência domina, embora continuem colhendo ganhos substanciais em MAD (especialmente na escala 4×). Em todos os dispositivos, o batching de 12 pacotes reduz o tempo efetivo por pacote e melhora o throughput, mas deixa claro que 70–90 % do tempo dos delegates corresponde a copiar dados, como evidenciado nos gráficos de tempo e pela análise de granularidade da Seção 2.7. Assim, futuras otimizações devem priorizar buffers persistentes e pipelines de aquisição in-place para liberar os delegates de transferências redundantes e desbloquear ganhos adicionais de desempenho e eficiência energética.

4.6 Precisão observada

O mesmo `TfliteDelegatesInstrumentedTest` que serviu de gate foi estendido com a suíte `fft_precision_statistics_suite`. Ao executar `./gradlew :app:connectedAndroidTest -Pandroid.testInstrumentationRunnerArguments.precisionRepeats=5`, o teste percorre cada comprimento de 512 a 65 536 pontos cinco vezes, registra as diferenças em `precision_fft.csv` e imprime linhas como `FFT_PRECISION_STATS: len=65536 (5x) | maxRel avg=11.104% std=12.779% min=3.605% max=36.572% | meanRel avg=0.007% | rmse avg=2.004928`. As escalas até 16 k apresentaram médias inferiores a 3 % e desvios baixos; em 32 k e 65 k surgiram picos ocasionais (resposta direta da magnitude dos bins e do uso de FP32 em CPU), mas a média relativa permaneceu na casa de 0,007 %. Comprimentos acima de 65 k não foram incluídos porque os dispositivos não completaram as execuções por falta de memória. Esses registros explicam por que diferentes delegates exibem pequenas variações mesmo usando o mesmo `.tflite`: GPU e NNAPI podem forçar FP16 na transferência, alguns bins passam a utilizar erro absoluto (guard rail de 1e3) e, sobretudo, todo o grafo roda no CPU via XNNPACK, de modo que qualquer oscilação deriva de cópias e arredondamentos em cada backend.
