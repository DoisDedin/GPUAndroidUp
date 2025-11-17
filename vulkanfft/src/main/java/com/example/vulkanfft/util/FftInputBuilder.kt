package com.example.vulkanfft.util

import kotlin.math.sqrt

/**
 * Converte as janelas de acelerômetro inteiras em vetores float prontos para o
 * pipeline de FFT e calcula pesos dinâmicos baseados na energia da janela.
 */
object FftInputBuilder {

    fun fromAccelerometer(
        batch: AccelerometerBatchGenerator.AccelerometerBatch,
        signalLength: Int
    ): FftProcessorInput {
        val numSensors = batch.sensors.size
        require(numSensors > 0) { "Batch vazio" }
        require(signalLength > 0) { "signalLength inválido" }
        val freqBins = signalLength / 2 + 1

        val samples = Array(numSensors) { sensorIndex ->
            val sensor = batch.sensors[sensorIndex]
            require(sensor.x.size >= signalLength) {
                "Sensor $sensorIndex não possui $signalLength amostras"
            }
            FloatArray(signalLength) { sampleIndex ->
                magnitude(
                    sensor.x[sampleIndex],
                    sensor.y[sampleIndex],
                    sensor.z[sampleIndex]
                )
            }
        }

        val weights = Array(numSensors) { sensorIndex ->
            val window = samples[sensorIndex]
            val energy = window.sum()
            val normalizedEnergy = energy / signalLength
            FloatArray(freqBins) { bin ->
                val slope = 0.0025f * bin
                1f + normalizedEnergy * ENERGY_SCALE + slope
            }
        }

        return FftProcessorInput(samples, weights)
    }

    private fun magnitude(x: Int, y: Int, z: Int): Float {
        val sum = x.toLong() * x + y.toLong() * y + z.toLong() * z
        return sqrt(sum.toDouble()).toFloat()
    }

    data class FftProcessorInput(
        val samples: Array<FloatArray>,
        val weights: Array<FloatArray>
    )

    private const val ENERGY_SCALE = 0.00001f
}
