package com.example.vulkanfft.util

import kotlin.math.PI
import kotlin.math.sin
import kotlin.random.Random

/**
 * Responsável por gerar janelas grandes (ex.: 4096 amostras) de dados de acelerômetro
 * para múltiplos sensores simultâneos. O formato retornado replica o input do MAD
 * (`Array<IntArray>` com timestamps, X, Y e Z) e pode ser reutilizado pelo pipeline
 * de FFT convertendo-o para magnitudes float.
 */
object AccelerometerBatchGenerator {

    fun generate(
        numSensors: Int,
        samplesPerSensor: Int,
        seed: Long = 42L
    ): AccelerometerBatch {
        require(numSensors > 0) { "numSensors precisa ser > 0" }
        require(samplesPerSensor > 0) { "samplesPerSensor precisa ser > 0" }

        val sensors = Array(numSensors) { sensorIndex ->
            val random = Random(seed + sensorIndex)
            val timestamps = IntArray(samplesPerSensor) { sampleIndex ->
                val base = sensorIndex * samplesPerSensor * TIMESTAMP_STEP
                base + sampleIndex * TIMESTAMP_STEP
            }
            val x = generateAxis(random, samplesPerSensor, sensorIndex, 0.8f)
            val y = generateAxis(random, samplesPerSensor, sensorIndex, 1.1f)
            val z = generateAxis(random, samplesPerSensor, sensorIndex, 1.3f)
            SensorData(
                timestamps = timestamps,
                x = x,
                y = y,
                z = z
            )
        }

        return AccelerometerBatch(sensors)
    }

    private fun generateAxis(
        random: Random,
        length: Int,
        sensorIndex: Int,
        amplitude: Float
    ): IntArray {
        val baseFrequency = 0.2f + sensorIndex * 0.05f
        val noiseScale = (sensorIndex + 1) * 0.3f
        return IntArray(length) { sampleIndex ->
            val t = sampleIndex / length.toFloat()
            val wave =
                sin(2f * PI.toFloat() * baseFrequency * t) +
                    0.5f * sin(2f * PI.toFloat() * baseFrequency * 2f * t)
            val noise = (random.nextFloat() - 0.5f) * noiseScale
            (amplitude * wave * ACC_SCALE + noise).toInt()
        }
    }

    data class AccelerometerBatch(
        val sensors: Array<SensorData>
    )

    data class SensorData(
        val timestamps: IntArray,
        val x: IntArray,
        val y: IntArray,
        val z: IntArray
    ) {
        fun asMadInput(): Array<IntArray> = arrayOf(timestamps, x, y, z)
    }

    private const val TIMESTAMP_STEP = 20 // ms ~50 Hz
    private const val ACC_SCALE = 1024f
}
