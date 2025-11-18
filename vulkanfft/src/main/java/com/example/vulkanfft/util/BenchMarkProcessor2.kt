package com.example.vulkanfft.util

import android.util.Log
import kotlin.math.pow
import kotlin.math.sqrt

class BenchmarkProcessor2(
    private val processorFactory: () -> StatModelProcessor,
    private val repetitions: Int = 30
) {

    private val tag = "BenchmarkProcessor2"

    suspend fun runBenchmark(input: Array<IntArray>): BenchmarkResult {
        Log.d(tag, "===== Início do Benchmark Estatístico =====")

        val timingSamples = mutableListOf<InferenceTiming>()

        repeat(repetitions) {
            val processor = processorFactory()

            val result = processor.process(input)

            processor.close()

            val durationMs = result.timing.totalMs
            timingSamples.add(result.timing)

            Log.d(
                tag,
                "Execução ${it + 1}: total=${"%.3f".format(durationMs)} ms " +
                    "(transfer=${"%.3f".format(result.timing.transferMs)} ms | " +
                    "proc=${"%.3f".format(result.timing.computeMs)} ms) - Saída[0]: ${result.output[0]}"
            )
        }

        val durations = timingSamples.map { it.totalMs }
        val mean = durations.average()
        val stdDev = sqrt(durations.map { (it - mean).pow(2) }.average())
        val min = durations.minOrNull() ?: 0.0
        val max = durations.maxOrNull() ?: 0.0

        Log.d(tag, "===== Resultados do Benchmark =====")
        Log.d(tag, "Média: ${"%.3f".format(mean)} ms")
        Log.d(tag, "Desvio Padrão: ${"%.3f".format(stdDev)} ms")
        Log.d(tag, "Min: ${"%.3f".format(min)} ms")
        Log.d(tag, "Max: ${"%.3f".format(max)} ms")
        Log.d(tag, "====================================")

        return BenchmarkResult(
            mean = mean,
            stdDev = stdDev,
            min = min,
            max = max,
            durations = durations,
            timings = timingSamples
        )
    }

    data class BenchmarkResult(
        val mean: Double,
        val stdDev: Double,
        val min: Double,
        val max: Double,
        val durations: List<Double>,
        val timings: List<InferenceTiming>
    )
}
