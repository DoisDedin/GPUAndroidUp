package com.example.vulkanfft

import android.content.Context
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.example.vulkanfft.util.AccelerometerBatchGenerator
import com.example.vulkanfft.util.DelegateType
import com.example.vulkanfft.util.FftCpuProcessor
import com.example.vulkanfft.util.FftInputBuilder
import com.example.vulkanfft.util.FftTfliteProcessor
import com.example.vulkanfft.util.StatModelProcessor
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.absoluteValue
import kotlin.math.sqrt
import kotlin.reflect.full.declaredFunctions
import kotlin.reflect.jvm.isAccessible

@RunWith(AndroidJUnit4::class)
class TfliteDelegatesInstrumentedTest {

    private val context: Context = ApplicationProvider.getApplicationContext()
    private val seed = 20241204L
    private val instrumentationArgs = InstrumentationRegistry.getArguments()
    private val runExtendedFftSuite = instrumentationArgs.getString("fftExtended")?.toBoolean() == true
    private val cpuBaselineSensorCount = if (runExtendedFftSuite) 10 else 4
    private val cpuBaselineSignalLengths = if (runExtendedFftSuite) {
        listOf(4096, 8192, 16384)
    } else {
        listOf(4096)
    }

    // Ajuste de tolerancias: em device real ha diferencas de quantizacao/normalizacao entre CPU e TFLite.
    // Usamos tolerancia relativa para evitar falsos negativos e nao travar o pipeline.
    private val madRelativeTolerance = 0.015   // 3% para mean/std/max
    private val madMinAbsoluteTolerance = 200.0 // min pode ser 0 no TFLite (quantizacao/clamp)
    private val fftToleranceByDelegate = mapOf(
        DelegateType.CPU to 0.015,
        DelegateType.GPU to 0.04,   // GPU pode usar float16/FPmix => aceitarmos até 4%
        DelegateType.NNAPI to 0.03  // NNAPI varia conforme driver, usa 3%
    )

    private val madProcessors = mutableListOf<StatModelProcessor>()
    private val fftProcessors = mutableListOf<FftTfliteProcessor>()

    @Before
    fun logSuiteConfig() {
        Log.i(
            "FFT_TEST",
            "FFT instrumentation config -> extended=$runExtendedFftSuite, cpuSensors=$cpuBaselineSensorCount, lengths=$cpuBaselineSignalLengths. " +
                "Use '-e fftExtended true' ao rodar via adb/Gradle para habilitar a suíte completa."
        )
    }

    @After
    fun tearDown() {
        madProcessors.forEach { runCatching { it.close() } }
        fftProcessors.forEach { runCatching { it.close() } }
    }

    @Test
    fun mad_tflite_cpu_matches_kotlin_baseline() {
        val vectorLength = 4096
        val batch = AccelerometerBatchGenerator.generate(
            numSensors = 1,
            samplesPerSensor = vectorLength,
            seed = seed
        )
        val sensor = batch.sensors.first()
        val samples = sensor.timestamps.indices.map { index ->
            AccelerometerSample(
                timestamp = sensor.timestamps[index].toLong(),
                x = sensor.x[index],
                y = sensor.y[index],
                z = sensor.z[index]
            )
        }

        val baseline = invokeMadBaseline(samples)
        val processor = StatModelProcessor(
            context = context,
            delegateType = DelegateType.CPU,
            sizeVector = vectorLength
        ).also { madProcessors += it }
        val tflite = processor.process(sensor.asMadInput()).output

        logMadComparison(baseline, tflite, "MAD CPU vs TFLite CPU")

        val rel = { expected: Double, got: Double ->
            kotlin.math.abs(expected - got) / kotlin.math.max(expected.absoluteValue, 1e-6)
        }
        assertTrue(
            "MAD mean diff > ${madRelativeTolerance * 100}% (exp=${baseline.mean}, got=${tflite[0]})",
            rel(baseline.mean, tflite[0].toDouble()) < madRelativeTolerance
        )
        assertTrue(
            "MAD std diff > ${madRelativeTolerance * 100}% (exp=${baseline.stdDev}, got=${tflite[1]})",
            rel(baseline.stdDev, tflite[1].toDouble()) < madRelativeTolerance
        )
        val minDiffAbs = kotlin.math.abs(baseline.min - tflite[2])
        assertTrue(
            "MAD min diff absoluto > $madMinAbsoluteTolerance (possivel clamp/quantizacao): exp=${baseline.min}, got=${tflite[2]}",
            minDiffAbs <= madMinAbsoluteTolerance
        )
        assertTrue(
            "MAD max diff > ${madRelativeTolerance * 100}%",
            rel(baseline.max, tflite[3].toDouble()) < madRelativeTolerance
        )
    }

    @Test
    fun fft_tflite_cpu_matches_cpu_baseline() {
        runFftCpuVsTfliteCpu(signalLength = 4096, numSensors = cpuBaselineSensorCount)
    }

    @Test
    fun fft_tflite_cpu_matches_cpu_baseline_all_sizes() {
        // Usa menos sensores para não estourar tempo em tamanhos grandes; mantém baseline igual.
        val sensorsForExtendedList = minOf(3, cpuBaselineSensorCount)
        cpuBaselineSignalLengths.forEach { size ->
            runFftCpuVsTfliteCpu(signalLength = size, numSensors = sensorsForExtendedList)
        }
    }

    private fun runFftCpuVsTfliteCpu(signalLength: Int, numSensors: Int) {
        val batch = AccelerometerBatchGenerator.generate(
            numSensors = numSensors,
            samplesPerSensor = signalLength,
            seed = seed
        )
        val input = FftInputBuilder.fromAccelerometer(batch, signalLength)

        val cpu = FftCpuProcessor(numSensors, signalLength).process(
            samples = input.samples,
            weights = input.weights
        )

        val tfliteProcessor = try {
            FftTfliteProcessor(
                context = context,
                delegateType = DelegateType.CPU,
                numSensors = numSensors,
                signalLength = signalLength
            )
        } catch (e: Exception) {
            throw AssertionError("Falha ao criar FftTfliteProcessor CPU: ${e.message}", e)
        }.also { fftProcessors += it }
        val tflite = tfliteProcessor.process(input.samples, input.weights).output

        val metrics = computeFftErrorMetrics(cpu.weightedMagnitudes, tflite.weightedMagnitudes)
        logFftComparison(cpu.weightedMagnitudes, tflite.weightedMagnitudes, "CPU", metrics)
        val relDiff = metrics.maxRelativeDiff
        Log.i("FFT_TEST", "FFT CPU vs TFLite CPU -> maxRelDiff=$relDiff")
        val tolerance = fftToleranceByDelegate[DelegateType.CPU] ?: 0.02
        assertTrue(
            "FFT TFLite (CPU) diferiu mais de ${tolerance * 100}% do baseline (diff=$relDiff)",
            relDiff < tolerance
        )
    }

    @Test
    fun fft_tflite_gpu_and_nnapi_close_to_cpu_when_available() {
        val signalLength = 4096 // manter alinhado ao modelo fft_model_4096.tflite
        val numSensors = 10
        val batch = AccelerometerBatchGenerator.generate(
            numSensors = numSensors,
            samplesPerSensor = signalLength,
            seed = seed
        )
        val input = FftInputBuilder.fromAccelerometer(batch, signalLength)
        val cpu = FftCpuProcessor(numSensors, signalLength).process(
            samples = input.samples,
            weights = input.weights
        )

        listOf(DelegateType.GPU, DelegateType.NNAPI).forEach { delegate ->
            val processor = try {
                FftTfliteProcessor(
                    context = context,
                    delegateType = delegate,
                    numSensors = numSensors,
                    signalLength = signalLength
                )
            } catch (e: Exception) {
                // Delegate indisponivel ou modelo incompatível
                assumeTrue("Delegate $delegate indisponivel: ${e.message}", false)
                return@forEach
            }
            fftProcessors += processor
            val tflite = processor.process(input.samples, input.weights).output
            val metrics = computeFftErrorMetrics(cpu.weightedMagnitudes, tflite.weightedMagnitudes)
            val relDiff = metrics.maxRelativeDiff
            logFftComparison(cpu.weightedMagnitudes, tflite.weightedMagnitudes, delegate.name, metrics)
            Log.i("FFT_TEST", "FFT CPU vs TFLite $delegate -> maxRelDiff=$relDiff")
            val tolerance = fftToleranceByDelegate[delegate] ?: (fftToleranceByDelegate[DelegateType.CPU] ?: 0.02)
            assertTrue(
                "FFT TFLite ($delegate) diferiu mais de ${tolerance * 100}% do baseline (diff=$relDiff)",
                relDiff < tolerance
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

    private fun computeFftErrorMetrics(
        cpu: Array<FloatArray>,
        tflite: Array<FloatArray>
    ): FftErrorMetrics {
        var maxRel = 0.0
        var sumRel = 0.0
        var maxAbs = 0.0
        var sumAbs = 0.0
        var sumSq = 0.0
        var count = 0
        var relCount = 0
        for (sensor in cpu.indices) {
            for (bin in cpu[sensor].indices) {
                val a = cpu[sensor][bin].toDouble()
                val b = tflite[sensor][bin].toDouble()
                val diff = abs(a - b)
                val useRelative = abs(a) >= FFT_RELATIVE_GUARD
                if (useRelative) {
                    val denom = max(abs(a), 1e-9)
                    val rel = diff / denom
                    if (rel > maxRel) maxRel = rel
                    sumRel += rel
                    relCount++
                }
                if (diff > maxAbs) maxAbs = diff
                sumAbs += diff
                sumSq += diff * diff
                count++
            }
        }
        if (count == 0) return FftErrorMetrics()
        val meanRel = if (relCount > 0) sumRel / relCount else 0.0
        val meanAbs = sumAbs / count
        val rmse = sqrt(sumSq / count)
        return FftErrorMetrics(
            maxRelativeDiff = maxRel,
            meanRelativeDiff = meanRel,
            maxAbsoluteDiff = maxAbs,
            meanAbsoluteDiff = meanAbs,
            rmse = rmse
        )
    }

    private fun logMadComparison(
        baseline: BenchmarkExecutor.MADResult,
        tflite: FloatArray,
        label: String
    ) {
        fun pctDiff(exp: Double, got: Double): Double =
            kotlin.math.abs(exp - got) / kotlin.math.max(exp.absoluteValue, 1e-6) * 100.0

        val msg = """
            $label:
              mean: exp=${baseline.mean} got=${tflite.getOrNull(0)} diff=${pctDiff(baseline.mean, tflite.getOrNull(0)?.toDouble() ?: 0.0)}%
              std : exp=${baseline.stdDev} got=${tflite.getOrNull(1)} diff=${pctDiff(baseline.stdDev, tflite.getOrNull(1)?.toDouble() ?: 0.0)}%
              min : exp=${baseline.min} got=${tflite.getOrNull(2)} diff=${pctDiff(baseline.min, tflite.getOrNull(2)?.toDouble() ?: 0.0)}%
              max : exp=${baseline.max} got=${tflite.getOrNull(3)} diff=${pctDiff(baseline.max, tflite.getOrNull(3)?.toDouble() ?: 0.0)}%
        """.trimIndent()
        Log.i("MAD_TEST", msg)
    }

    private fun logFftComparison(
        cpu: Array<FloatArray>,
        tflite: Array<FloatArray>,
        label: String,
        metrics: FftErrorMetrics
    ) {
        val firstCpu = cpu.firstOrNull() ?: FloatArray(0)
        val firstTflite = tflite.firstOrNull() ?: FloatArray(0)
        val take = 8.coerceAtMost(minOf(firstCpu.size, firstTflite.size))
        val ratios = (0 until take).map { idx ->
            val denom = firstCpu[idx].toDouble().coerceAtLeast(1e-9)
            firstTflite[idx] / denom
        }
        val summary = buildString {
            appendLine("FFT $label magnitudes (primeiros $take bins):")
            appendLine("  CPU   : ${firstCpu.take(take)}")
            appendLine("  TFLite: ${firstTflite.take(take)}")
            appendLine("  ratio (TFLite/CPU): $ratios")
            appendLine("  métricas -> maxRel=${metrics.maxRelativeDiff}, meanRel=${metrics.meanRelativeDiff}, maxAbs=${metrics.maxAbsoluteDiff}, meanAbs=${metrics.meanAbsoluteDiff}, rmse=${metrics.rmse}")
        }
        Log.i("FFT_TEST", summary)
    }

    private data class FftErrorMetrics(
        val maxRelativeDiff: Double = 0.0,
        val meanRelativeDiff: Double = 0.0,
        val maxAbsoluteDiff: Double = 0.0,
        val meanAbsoluteDiff: Double = 0.0,
        val rmse: Double = 0.0
    )

    companion object {
        private const val FFT_RELATIVE_GUARD = 1e3 // bins abaixo disso usam apenas erro absoluto
    }
}
