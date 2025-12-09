package com.example.vulkanfft.util

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class AccelerometerBatchGeneratorTest {

    @Test
    fun `generator produces monotonic timestamps and full axes`() {
        val batch = AccelerometerBatchGenerator.generate(
            numSensors = 3,
            samplesPerSensor = 32,
            seed = 7L
        )

        assertEquals(3, batch.sensors.size)
        batch.sensors.forEachIndexed { index, sensor ->
            assertEquals(32, sensor.timestamps.size)
            assertEquals(32, sensor.x.size)
            assertEquals(32, sensor.y.size)
            assertEquals(32, sensor.z.size)
            val diffs = sensor.timestamps.asList().zipWithNext { a, b -> b - a }
            assertTrue("Sensor $index possui timestamps não crescentes", diffs.all { it > 0 })
        }
    }

    @Test
    fun `generator handles half million samples`() {
        val samples = 526_000
        val batch = AccelerometerBatchGenerator.generate(
            numSensors = 1,
            samplesPerSensor = samples,
            seed = 99L
        )

        val sensor = batch.sensors.single()
        assertEquals(samples, sensor.timestamps.size)
        assertEquals(samples, sensor.x.size)
        assertEquals(samples, sensor.y.size)
        assertEquals(samples, sensor.z.size)
        // Garante que o sinal possui variação nas primeiras 10k amostras.
        var transitions = 0
        for (index in 1 until 10_000) {
            if (sensor.x[index] != sensor.x[index - 1]) {
                transitions++
                if (transitions > 500) break
            }
        }
        assertTrue("Variação insuficiente para $samples amostras", transitions > 500)
    }
}
