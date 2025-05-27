import android.util.Log
import kotlin.math.sqrt

class FFTProcessor {

    private val tag = "DistanceProcessor"

    fun executeDistanceCalculation(input: DoubleArray): DoubleArray {
        Log.d(tag, "====== Início do cálculo de distâncias ======")
        val startTime = System.currentTimeMillis()

        Log.d(tag, "Input: ${input.joinToString()}")

        val result = try {
            val output = calculateDistanceBetweenPoints(input)
            output
        } catch (e: Exception) {
            Log.e(tag, "Erro durante o cálculo: ${e.message}")
            DoubleArray(0)
        }

        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime

        Log.d(tag, "Resultado: ${result.joinToString()}")
        Log.d(tag, "Tempo total (ms): $duration")

        if (validateResult(result)) {
            Log.d(tag, "Resultado validado com sucesso!")
        } else {
            Log.e(tag, "ERRO: Resultado inválido!")
        }

        Log.d(tag, "====== Fim do cálculo de distâncias ======")

        return result
    }

    fun calculateTotalDistanceCPU(input: DoubleArray): Double {
        Log.d(tag, "====== Início do somatório de distâncias (CPU) ======")
        val startTime = System.currentTimeMillis()

        val distances = calculateDistanceBetweenPoints(input)
        val totalDistance = distances.sum()

        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime

        Log.d(tag, "Total Distance: $totalDistance")
        Log.d(tag, "Tempo total (ms): $duration")

        Log.d(tag, "====== Fim do somatório (CPU) ======")

        return totalDistance
    }

    private fun calculateDistanceBetweenPoints(input: DoubleArray): DoubleArray {
        val distances = DoubleArray(input.size - 1) { i ->
            kotlin.math.abs(input[i + 1] - input[i])
        }
        return distances
    }

    private fun validateResult(result: DoubleArray): Boolean {
        if (result.isEmpty()) return false
        return result.all { !it.isNaN() && !it.isInfinite() }
    }
}