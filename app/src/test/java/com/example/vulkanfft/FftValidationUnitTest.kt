package com.example.vulkanfft

import com.example.vulkanfft.util.FftCpuProcessor
import com.example.vulkanfft.util.FftNormalization
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Testes locais de coerencia da FFT em CPU.
 * - Compara a saida do FftCpuProcessor com um DFT manual simples (tamanho pequeno).
 * - Valida aplicacao de pesos (weightedMagnitudes).
 *
 * Comparacao contra TFLite fica no teste instrumentado.
 */
class FftValidationUnitTest {

    @Test
    fun fft_cpu_matches_manual_dft_for_small_signal() {
        val numSensors = 1
        val signalLength = 8
        val processor = FftCpuProcessor(numSensors, signalLength)

        val samples = arrayOf(
            FloatArray(signalLength) { i -> cos(2 * PI * i / signalLength).toFloat() }
        )
        val weights = Array(numSensors) { FloatArray(signalLength / 2 + 1) { 1f } }

        val result = processor.process(samples, weights)
        val manual = manualDft(samples[0])

        manual.forEachIndexed { index, mag ->
            assertEquals("Magnitude bin $index", mag, result.magnitudes[0][index], 1e-3f)
        }
    }

    @Test
    fun fft_cpu_applies_weights() {
        val numSensors = 1
        val signalLength = 8
        val processor = FftCpuProcessor(numSensors, signalLength)
        val weights = Array(numSensors) { FloatArray(signalLength / 2 + 1) { i -> (i + 1).toFloat() } }
        val samples = arrayOf(
            FloatArray(signalLength) { i -> sin(2 * PI * i / signalLength).toFloat() }
        )

        val result = processor.process(samples, weights)
        val unweighted = result.magnitudes[0]
        val weighted = result.weightedMagnitudes[0]

        val expected = FloatArray(unweighted.size) { i -> unweighted[i] * (i + 1) }
        assertArrayEquals(expected, weighted, 1e-4f)
    }

    @Test
    fun fft_cpu_respects_normalization_modes() {
        val numSensors = 1
        val signalLength = 16
        val baseSamples = arrayOf(
            FloatArray(signalLength) { i -> cos(2 * PI * 3 * i / signalLength).toFloat() }
        )
        val weights = Array(numSensors) { FloatArray(signalLength / 2 + 1) { 1f } }
        val manual = manualDft(baseSamples[0])

        val none = FftCpuProcessor(numSensors, signalLength, FftNormalization.NONE)
            .process(baseSamples, weights).magnitudes[0]
        assertArrayEquals("Sem normalização deveria coincidir com DFT bruto", manual, none, 1e-3f)

        val byN = FftCpuProcessor(numSensors, signalLength, FftNormalization.BY_SIGNAL_LENGTH)
            .process(baseSamples, weights).magnitudes[0]
        val expectedByN = manual.map { it / signalLength }.toFloatArray()
        assertArrayEquals("Normalização por N inválida", expectedByN, byN, 1e-3f)

        val bySqrt = FftCpuProcessor(numSensors, signalLength, FftNormalization.BY_SQRT_SIGNAL_LENGTH)
            .process(baseSamples, weights).magnitudes[0]
        val expectedBySqrt = manual.map { it / kotlin.math.sqrt(signalLength.toFloat()) }.toFloatArray()
        assertArrayEquals("Normalização por sqrt(N) inválida", expectedBySqrt, bySqrt, 1e-3f)
    }

    private fun manualDft(signal: FloatArray): FloatArray {
        val n = signal.size
        val bins = n / 2 + 1
        val magnitudes = FloatArray(bins)
        for (k in 0 until bins) {
            var real = 0.0
            var imag = 0.0
            for (t in 0 until n) {
                val angle = -2.0 * PI * k * t / n
                real += signal[t] * cos(angle)
                imag += signal[t] * sin(angle)
            }
            magnitudes[k] = kotlin.math.sqrt(real * real + imag * imag).toFloat()
        }
        return magnitudes
    }
}
