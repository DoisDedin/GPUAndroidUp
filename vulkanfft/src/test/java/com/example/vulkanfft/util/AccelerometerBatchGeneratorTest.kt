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
            val diffs = sensor.timestamps.zipWithNext { a, b -> b - a }
            assertTrue("Sensor $index possui timestamps não crescentes", diffs.all { it > 0 })
        }
    }
}
