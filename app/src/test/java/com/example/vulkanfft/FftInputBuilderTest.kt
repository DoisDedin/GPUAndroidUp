package com.example.vulkanfft

import com.example.vulkanfft.util.AccelerometerBatchGenerator
import com.example.vulkanfft.util.FftInputBuilder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class FftInputBuilderTest {

    @Test
    fun samples_and_weights_match_expected_values() {
        val signalLength = 4
        val batch = AccelerometerBatchGenerator.AccelerometerBatch(
            sensors = arrayOf(
                AccelerometerBatchGenerator.SensorData(
                    timestamps = intArrayOf(0, 20, 40, 60),
                    x = intArrayOf(3, 0, 0, 1),
                    y = intArrayOf(4, 0, 8, 2),
                    z = intArrayOf(0, 6, 0, 2)
                )
            )
        )

        val input = FftInputBuilder.fromAccelerometer(batch, signalLength)
        val samples = input.samples.first()
        val weights = input.weights.first()

        val expectedMagnitudes = floatArrayOf(5f, 6f, 8f, 3f)
        assertArrayEquals(expectedMagnitudes, samples, 1e-6f)

        val normalizedEnergy = expectedMagnitudes.sum() / signalLength
        val energyContribution = normalizedEnergy * 0.00001f
        val expectedWeights = floatArrayOf(
            1f + energyContribution,
            1f + energyContribution + 0.0025f,
            1f + energyContribution + 0.005f
        )
        assertArrayEquals(expectedWeights, weights, 1e-6f)
    }

    @Test
    fun throws_when_sensor_has_not_enough_samples() {
        val batch = AccelerometerBatchGenerator.AccelerometerBatch(
            sensors = arrayOf(
                AccelerometerBatchGenerator.SensorData(
                    timestamps = intArrayOf(0, 20),
                    x = intArrayOf(1, 1),
                    y = intArrayOf(0, 0),
                    z = intArrayOf(0, 0)
                )
            )
        )

        assertThrows(IllegalArgumentException::class.java) {
            FftInputBuilder.fromAccelerometer(batch, signalLength = 4)
        }
    }
}
