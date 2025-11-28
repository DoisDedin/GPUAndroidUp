package com.example.vulkanfft

import com.example.vulkanfft.util.AccelerometerBatchGenerator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import kotlin.math.round
import kotlin.reflect.full.declaredFunctions
import kotlin.reflect.jvm.isAccessible

/**
 * Valida a coerencia do MAD em CPU (Kotlin puro) usando a implementacao de producao
 * (BenchmarkExecutor.getMAD via reflexao) e uma checagem manual.
 *
 * Observacao: comparacao com TFLite fica no teste instrumentado, pois precisa de assets
 * e delegates Android.
 */
class MadValidationUnitTest {

    private val seed = 1234L

    @Test
    fun mad_baseline_matches_manual_stats() {
        val samples = buildSamples(vectorLength = 256, seed = seed)
        val baseline = invokeMadBaseline(samples)
        val manual = manualMad(samples)

        assertNotNull(baseline)
        assertEquals(manual.mean, baseline.mean, 1e-6)
        assertEquals(manual.stdDev, baseline.stdDev, 1e-6)
        assertEquals(manual.min, baseline.min, 1e-6)
        assertEquals(manual.max, baseline.max, 1e-6)
    }

    @Test
    fun mad_baseline_is_deterministic_with_fixed_seed() {
        val a = invokeMadBaseline(buildSamples(vectorLength = 512, seed = seed))
        val b = invokeMadBaseline(buildSamples(vectorLength = 512, seed = seed))
        assertEquals(a.mean, b.mean, 0.0)
        assertEquals(a.stdDev, b.stdDev, 0.0)
        assertEquals(a.min, b.min, 0.0)
        assertEquals(a.max, b.max, 0.0)
    }

    private fun buildSamples(vectorLength: Int, seed: Long): List<AccelerometerSample> {
        val batch = AccelerometerBatchGenerator.generate(
            numSensors = 1,
            samplesPerSensor = vectorLength,
            seed = seed
        )
        val sensor = batch.sensors.first()
        return sensor.timestamps.indices.map { index ->
            AccelerometerSample(
                timestamp = sensor.timestamps[index].toLong(),
                x = sensor.x[index],
                y = sensor.y[index],
                z = sensor.z[index]
            )
        }
    }

    private fun invokeMadBaseline(samples: List<AccelerometerSample>): BenchmarkExecutor.MADResult {
        val method = BenchmarkExecutor::class.declaredFunctions.first { it.name == "getMAD" }
        method.isAccessible = true
        val executor = BenchmarkExecutor()
        @Suppress("UNCHECKED_CAST")
        return method.call(executor, samples) as BenchmarkExecutor.MADResult
    }

    private fun manualMad(samples: List<AccelerometerSample>): BenchmarkExecutor.MADResult {
        val magnitudeByTime = samples.map { sample ->
            val magnitude = kotlin.math.sqrt(
                (sample.x * sample.x + sample.y * sample.y + sample.z * sample.z).toDouble()
            )
            sample.timestamp to magnitude
        }
        val windowSizeMillis = 5_000L
        val startTs = magnitudeByTime.first().first
        val blockAverages = mutableListOf<Double>()
        var currentStart = startTs
        while (true) {
            val end = currentStart + windowSizeMillis
            val window = magnitudeByTime.filter { it.first in currentStart until end }.map { it.second }
            if (window.isEmpty()) break
            blockAverages += window.average()
            currentStart = end
            if (currentStart > magnitudeByTime.last().first) break
        }
        val mean = blockAverages.average()
        val variance = blockAverages.map { (it - mean) * (it - mean) }.average()
        val stdDev = kotlin.math.sqrt(variance)
        val min = blockAverages.minOrNull() ?: 0.0
        val max = blockAverages.maxOrNull() ?: 0.0
        return BenchmarkExecutor.MADResult(
            mean = round1e6(mean),
            stdDev = round1e6(stdDev),
            min = min,
            max = max
        )
    }

    private fun round1e6(value: Double): Double {
        val scale = 1_000_000.0
        return kotlin.math.round(value * scale) / scale
    }
}
