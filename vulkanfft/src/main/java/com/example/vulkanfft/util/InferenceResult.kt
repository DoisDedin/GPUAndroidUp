package com.example.vulkanfft.util

/**
 * Representa o tempo gasto para preparar as entradas (transferência para a memória
 * do delegate) e o tempo de processamento em si.
 */
data class InferenceTiming(
    val transferMs: Double,
    val computeMs: Double
) {
    val totalMs: Double get() = transferMs + computeMs
}

/**
 * Encapsula a saída do modelo/processador junto com as métricas de tempo coletadas.
 */
data class InferenceResult<T>(
    val output: T,
    val timing: InferenceTiming
)
