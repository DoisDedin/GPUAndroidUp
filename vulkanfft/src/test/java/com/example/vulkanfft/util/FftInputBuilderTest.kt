package com.example.vulkanfft.util

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FftInputBuilderTest {

    @Test
    fun `builder converts accelerometer axes into magnitudes and weights`() {
        val sensor = AccelerometerBatchGenerator.SensorData(
            timestamps = IntArray(8) { it * 20 },
            x = IntArray(8) { it },
            y = IntArray(8) { 0 },
            z = IntArray(8) { 0 }
        )
        val batch = AccelerometerBatchGenerator.AccelerometerBatch(arrayOf(sensor))

        val input = FftInputBuilder.fromAccelerometer(batch, signalLength = 8)

        assertEquals(1, input.samples.size)
        assertEquals(8, input.samples[0].size)
        assertEquals(0f, input.samples[0][0], 1e-3f)
        assertEquals(5.0, input.samples[0][5].toDouble(), 1e-3)

        val weights = input.weights[0]
        assertEquals(5, weights.size)
        assertTrue(weights.all { it > 1f })
        assertTrue(weights.zipWithNext().all { (a, b) -> b >= a })
    }
}
