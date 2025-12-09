package com.example.vulkanfft.util

import android.util.Log
import org.jtransforms.fft.FloatFFT_1D
import kotlin.math.sqrt

/**
 * Implementação CPU que replica o pipeline do TFLite: calcula FFT real (via JTransforms)
 * por sensor, aplica a mesma normalização escolhida e depois multiplica pelos pesos
 * de frequência. Serve como baseline determinístico para os testes instrumentados.
 *
 * @param numSensors Número de sensores processados simultaneamente.
 * @param signalLength Comprimento da janela por sensor.
 * @param normalization Convenção de normalização para alinhar com tf.signal.rfft.
 */
class FftCpuProcessor(
    private val numSensors: Int,
    private val signalLength: Int,
    private val normalization: FftNormalization = FftNormalization.NONE
) {

    private val freqBins = signalLength / 2 + 1
    private val fft = FloatFFT_1D(signalLength.toLong())
    // tf.signal.rfft não aplica normalização; mantemos o mesmo padrão em NONE.
    private val normalizationFactor = normalization.factor(signalLength)

    init {
        require(signalLength > 1) { "signalLength precisa ser > 1" }
        require(numSensors > 0) { "numSensors precisa ser > 0" }
    }

    fun process(
        samples: Array<FloatArray>,
        weights: Array<FloatArray>
    ): FftResult {
        logInfo("Iniciando FFT CPU: ${samples.size} sensores × $signalLength amostras")
        require(samples.size == numSensors) {
            "Esperado $numSensors sensores, recebido ${samples.size}"
        }
        require(weights.size == numSensors) {
            "Esperado vetor de pesos por sensor ($numSensors), recebido ${weights.size}"
        }

        val complexSpectra = Array(numSensors) { Array(freqBins) { ComplexFloat(0f, 0f) } }

        val magnitudes = Array(numSensors) { FloatArray(freqBins) }
        val weightedMagnitudes = Array(numSensors) { FloatArray(freqBins) }

        samples.forEachIndexed { sensorIndex, signal ->
            require(signal.size == signalLength) {
                "Sensor #$sensorIndex deveria ter $signalLength amostras, mas tem ${signal.size}"
            }

            val scratch = FloatArray(2 * signalLength)
            System.arraycopy(signal, 0, scratch, 0, signalLength)
            val start = System.nanoTime()
            fft.realForwardFull(scratch)
            val durationMs = (System.nanoTime() - start) / 1_000_000.0
            logInfo("Sensor ${sensorIndex + 1}/$numSensors finalizado em ${"%.1f".format(durationMs)} ms")

            for (bin in 0 until freqBins) {
                val real = scratch[bin * 2] * normalizationFactor
                val imag = scratch[bin * 2 + 1] * normalizationFactor
                val magnitude = sqrt(real * real + imag * imag)

                complexSpectra[sensorIndex][bin] = ComplexFloat(real, imag)
                magnitudes[sensorIndex][bin] = magnitude
            }
        }

        complexSpectra.forEachIndexed { sensorIndex, _ ->
            val weightVector = weights[sensorIndex]
            require(weightVector.size == freqBins) {
                "Vetor de pesos do sensor #$sensorIndex deveria ter $freqBins posições"
            }

            for (binIndex in 0 until freqBins) {
                val magnitude = magnitudes[sensorIndex][binIndex]
                weightedMagnitudes[sensorIndex][binIndex] = magnitude * weightVector[binIndex]
            }
        }

        logInfo("FFT CPU concluída para todos os sensores.")
        return FftResult(
            complexSpectrum = complexSpectra,
            magnitudes = magnitudes,
            weightedMagnitudes = weightedMagnitudes
        )
    }

    /**
     * Versão “streaming” usada em testes instrumentados. Processa um sensor por vez,
     * reaproveitando buffers para reduzir picos de memória. O callback recebe os
     * arrays de magnitude/pesos do sensor atual e deve consumi-los imediatamente
     * (pois serão reutilizados no sensor seguinte).
     */
    fun processStreaming(
        samples: Array<FloatArray>,
        weights: Array<FloatArray>,
        consumer: (sensorIndex: Int, magnitudes: FloatArray, weightedMagnitudes: FloatArray) -> Unit
    ) {
        require(samples.size == numSensors) {
            "Esperado $numSensors sensores, recebido ${samples.size}"
        }
        require(weights.size == numSensors) {
            "Esperado vetor de pesos por sensor ($numSensors), recebido ${weights.size}"
        }

        val scratch = FloatArray(2 * signalLength)
        val freqBins = signalLength / 2 + 1
        val magnitudes = FloatArray(freqBins)
        val weighted = FloatArray(freqBins)

        samples.forEachIndexed { sensorIndex, signal ->
            require(signal.size == signalLength) {
                "Sensor #$sensorIndex deveria ter $signalLength amostras, mas tem ${signal.size}"
            }
            System.arraycopy(signal, 0, scratch, 0, signalLength)
            val start = System.nanoTime()
            fft.realForwardFull(scratch)
            logInfo("Sensor ${sensorIndex + 1}/$numSensors finalizado em ${"%.1f".format((System.nanoTime() - start) / 1_000_000.0)} ms")

            for (bin in 0 until freqBins) {
                val real = scratch[bin * 2] * normalizationFactor
                val imag = scratch[bin * 2 + 1] * normalizationFactor
                magnitudes[bin] = sqrt(real * real + imag * imag)
            }
            val weightVector = weights[sensorIndex]
            require(weightVector.size == freqBins) {
                "Vetor de pesos do sensor #$sensorIndex deveria ter $freqBins posições"
            }
            for (bin in 0 until freqBins) {
                weighted[bin] = magnitudes[bin] * weightVector[bin]
            }
            consumer(sensorIndex, magnitudes, weighted)
        }
    }

    companion object {
        private const val TAG = "FftCpuProcessor"
        private fun logInfo(message: String) {
            runCatching { Log.i(TAG, message) }.getOrElse {
                println("[$TAG] $message")
            }
        }
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

enum class FftNormalization {
    NONE,
    BY_SIGNAL_LENGTH,
    BY_SQRT_SIGNAL_LENGTH;

    fun factor(signalLength: Int): Float = when (this) {
        NONE -> 1f
        BY_SIGNAL_LENGTH -> 1f / signalLength.toFloat()
        BY_SQRT_SIGNAL_LENGTH -> 1f / sqrt(signalLength.toFloat())
    }
}
