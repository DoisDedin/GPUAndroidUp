package com.example.vulkanfft.util

import android.util.Log
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * CPU-only pipeline that mirrors the MAD processor architecture but executes
 * a weighted FFT per sensor channel. The implementation keeps the
 * transformation simple (direct DFT) so that we have a deterministic baseline
 * to benchmark against the TensorFlow Lite implementation.
 *
 * @param numSensors Number of sensor channels that will arrive simultaneously.
 * @param signalLength Number of samples per sensor (for this prototype we use 10).
 */
class FftCpuProcessor(
    private val numSensors: Int,
    private val signalLength: Int
) {

    private val freqBins = signalLength / 2 + 1

    init {
        require(signalLength > 1) { "signalLength precisa ser > 1" }
        require(numSensors > 0) { "numSensors precisa ser > 0" }
    }

    fun process(
        samples: Array<FloatArray>,
        weights: Array<FloatArray>
    ): FftResult {
        Log.i(TAG, "Iniciando FFT CPU: ${samples.size} sensores × $signalLength amostras")
        require(samples.size == numSensors) {
            "Esperado $numSensors sensores, recebido ${samples.size}"
        }
        require(weights.size == numSensors) {
            "Esperado vetor de pesos por sensor ($numSensors), recebido ${weights.size}"
        }

        val complexSpectra = Array(numSensors) { sensorIndex ->
            val signal = samples[sensorIndex]
            require(signal.size == signalLength) {
                "Sensor #$sensorIndex deveria ter $signalLength amostras, mas tem ${signal.size}"
            }
            val start = System.nanoTime()
            val spectrum = computeRealDft(signal, sensorIndex)
            val durationMs = (System.nanoTime() - start) / 1_000_000.0
            Log.i(TAG, "Sensor ${sensorIndex + 1}/$numSensors finalizado em ${"%.1f".format(durationMs)} ms")
            spectrum
        }

        val magnitudes = Array(numSensors) { FloatArray(freqBins) }
        val weightedMagnitudes = Array(numSensors) { FloatArray(freqBins) }

        complexSpectra.forEachIndexed { sensorIndex, bins ->
            val weightVector = weights[sensorIndex]
            require(weightVector.size == freqBins) {
                "Vetor de pesos do sensor #$sensorIndex deveria ter $freqBins posições"
            }

            bins.forEachIndexed { binIndex, complex ->
                val magnitude = complex.magnitude()
                magnitudes[sensorIndex][binIndex] = magnitude
                weightedMagnitudes[sensorIndex][binIndex] = magnitude * weightVector[binIndex]
            }
        }

        Log.i(TAG, "FFT CPU concluída para todos os sensores.")
        return FftResult(
            complexSpectrum = complexSpectra,
            magnitudes = magnitudes,
            weightedMagnitudes = weightedMagnitudes
        )
    }

    private fun computeRealDft(signal: FloatArray, sensorIndex: Int): Array<ComplexFloat> {
        val bins = Array(freqBins) { ComplexFloat(0f, 0f) }
        val twoPiByN = (2.0 * PI / signalLength).toFloat()
        val progressStep = (freqBins / 8).coerceAtLeast(1)

        for (k in 0 until freqBins) {
            var real = 0.0
            var imag = 0.0
            for (n in 0 until signalLength) {
                val angle = twoPiByN * k * n
                val sample = signal[n].toDouble()
                real += sample * cos(angle.toDouble())
                imag -= sample * sin(angle.toDouble())
            }
            bins[k] = ComplexFloat(real.toFloat(), imag.toFloat())
            if (k % progressStep == 0 || k == freqBins - 1) {
                val percent = ((k + 1) / freqBins.toDouble() * 100).toInt()
                Log.d(
                    TAG,
                    "Sensor ${sensorIndex + 1}/$numSensors -> FFT ${percent.coerceAtMost(100)}% concluída"
                )
            }
        }

        return bins
    }

    companion object {
        private const val TAG = "FftCpuProcessor"
    }
}

data class ComplexFloat(
    val real: Float,
    val imaginary: Float
) {
    fun magnitude(): Float = sqrt(real * real + imaginary * imaginary)
}

data class FftResult(
    val complexSpectrum: Array<Array<ComplexFloat>>,
    val magnitudes: Array<FloatArray>,
    val weightedMagnitudes: Array<FloatArray>
)
