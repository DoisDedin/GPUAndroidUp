package com.example.vulkanfft.util

import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertEquals
import org.junit.Test

class FftCpuProcessorTest {

    @Test
    fun `pure tone concentrates magnitude in target bin`() {
        val signalLength = 8
        val processor = FftCpuProcessor(numSensors = 1, signalLength = signalLength)
        val samples = Array(1) {
            FloatArray(signalLength) { sampleIndex ->
                sin(2f * PI.toFloat() * sampleIndex / signalLength)
            }
        }
        val weights = Array(1) { FloatArray(signalLength / 2 + 1) { 1f } }

        val result = processor.process(samples, weights)
        val magnitudes = result.magnitudes[0]

        assertEquals(0f, magnitudes[0], 1e-3f)
        assertEquals(4f, magnitudes[1], 1e-2f) // N/2 para seno puro
        assertEquals(0f, magnitudes[2], 1e-2f)
        assertEquals(0f, magnitudes[3], 1e-2f)
        assertEquals(0f, magnitudes[4], 1e-2f)
    }
}
