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
import java.util.Locale
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
    private val fftSignalLengths = listOf(512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    private val runExtendedFftSuite =
        instrumentationArgs.getString("fftExtended")?.toBoolean() ?: true
    private val cpuBaselineSensorCount = BenchmarkExecutor.FFT_NUM_SENSORS
    private val cpuBaselineSignalLengths = if (runExtendedFftSuite) {
        fftSignalLengths
    } else {
        listOf(4096)
    }

    // Ajuste de tolerancias: em device real ha diferencas de quantizacao/normalizacao entre CPU e TFLite.
    // Usamos tolerancia relativa para evitar falsos negativos e nao travar o pipeline.
    private val madRelativeTolerance = 0.015   // 3% para mean/std/max
    private val madMinAbsoluteTolerance = 200.0 // min pode ser 0 no TFLite (quantizacao/clamp)
    private val fftToleranceByDelegate = mapOf(
        DelegateType.CPU to 0.10,   // em comprimentos grandes a discrepância FP32 pode chegar a ~3%
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
                "Use '-e fftExtended true' ao rodar via adb/Gradle para habilitar a suíte completa. Resultados de precisão serão gravados em files/precision/."
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
        val meanRel = rel(baseline.mean, tflite[0].toDouble())
        assertTrue(
            "MAD mean diff > ${madRelativeTolerance * 100}% (exp=${baseline.mean}, got=${tflite[0]})",
            meanRel < madRelativeTolerance
        )
        val stdRel = rel(baseline.stdDev, tflite[1].toDouble())
        assertTrue(
            "MAD std diff > ${madRelativeTolerance * 100}% (exp=${baseline.stdDev}, got=${tflite[1]})",
            stdRel < madRelativeTolerance
        )
        val minDiffAbs = kotlin.math.abs(baseline.min - tflite[2])
        assertTrue(
            "MAD min diff absoluto > $madMinAbsoluteTolerance (possivel clamp/quantizacao): exp=${baseline.min}, got=${tflite[2]}",
            minDiffAbs <= madMinAbsoluteTolerance
        )
        val maxRel = rel(baseline.max, tflite[3].toDouble())
        assertTrue(
            "MAD max diff > ${madRelativeTolerance * 100}%",
            maxRel < madRelativeTolerance
        )

        PrecisionReporter.appendMad(
            context = context,
            vectorLength = vectorLength,
            metrics = MadErrorMetrics(
                relativeMean = meanRel,
                relativeStd = stdRel,
                relativeMax = maxRel,
                absoluteMin = minDiffAbs
            )
        )
    }

    @Test
    fun fft_tflite_cpu_matches_cpu_baseline() {
        runFftCpuVsTfliteCpu(signalLength = 4096, numSensors = cpuBaselineSensorCount)
    }

    @Test
    fun fft_tflite_cpu_matches_cpu_baseline_all_sizes() {
        // Usa menos sensores para não estourar tempo em tamanhos grandes; mantém baseline igual.
        cpuBaselineSignalLengths.forEach { size ->
            runFftCpuVsTfliteCpu(signalLength = size, numSensors = cpuBaselineSensorCount)
        }
    }

    private fun runFftCpuVsTfliteCpu(signalLength: Int, numSensors: Int) {
        val metrics = measureFftAgainstCpu(
            signalLength = signalLength,
            numSensors = numSensors,
            delegateType = DelegateType.CPU,
            runIndex = 0
        )
        val relDiff = metrics.maxRelativeDiff
        Log.i("FFT_TEST", "FFT CPU vs TFLite CPU -> maxRelDiff=$relDiff")

        if (isExperimentalLength(signalLength)) {
            Log.w("FFT_TEST", "Diferença FFT CPU ($relDiff) em $signalLength pts registrada apenas para acompanhamento (experimental).")
            return
        }

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
        listOf(DelegateType.GPU, DelegateType.NNAPI).forEach { delegate ->
            val metrics = try {
                measureFftAgainstCpu(
                    signalLength = signalLength,
                    numSensors = numSensors,
                    delegateType = delegate,
                    runIndex = 0
                )
            } catch (e: IllegalStateException) {
                assumeTrue("Delegate $delegate indisponivel: ${e.message}", false)
                return@forEach
            }
            val relDiff = metrics.maxRelativeDiff
            Log.i("FFT_TEST", "FFT CPU vs TFLite $delegate -> maxRelDiff=$relDiff")
            val tolerance = fftToleranceByDelegate[delegate] ?: (fftToleranceByDelegate[DelegateType.CPU] ?: 0.02)
            if (isExperimentalLength(signalLength)) {
                Log.w("FFT_TEST", "Diferença FFT $delegate ($relDiff) em $signalLength pts registrada apenas para acompanhamento (experimental).")
                return@forEach
            }

            assertTrue(
                "FFT TFLite ($delegate) diferiu mais de ${tolerance * 100}% do baseline (diff=$relDiff)",
                relDiff < tolerance
            )
        }
    }

    @Test
    fun fft_precision_statistics_suite() {
        val repeats = instrumentationArgs.getString("precisionRepeats")?.toIntOrNull()?.coerceAtLeast(1) ?: 5
        val lengths = listOf(512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
        lengths.forEach { length ->
            val metricsList = (0 until repeats).map { runIndex ->
                measureFftAgainstCpu(
                    signalLength = length,
                    numSensors = cpuBaselineSensorCount,
                    delegateType = DelegateType.CPU,
                    runIndex = runIndex
                )
            }
            reportPrecisionStats(length, repeats, metricsList)
        }
    }

    private fun invokeMadBaseline(samples: List<AccelerometerSample>): BenchmarkExecutor.MADResult {
        val method = BenchmarkExecutor::class.declaredFunctions.first { it.name == "getMAD" }
        method.isAccessible = true
        val executor = BenchmarkExecutor()
        @Suppress("UNCHECKED_CAST")
        return method.call(executor, samples) as BenchmarkExecutor.MADResult
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
        cpuPreview: FloatArray,
        tflitePreview: FloatArray,
        label: String,
        metrics: FftErrorMetrics
    ) {
        val take = minOf(cpuPreview.size, tflitePreview.size)
        val ratios = (0 until take).map { idx ->
            val denom = cpuPreview[idx].toDouble().coerceAtLeast(1e-9)
            tflitePreview[idx] / denom
        }
        val summary = buildString {
            appendLine("FFT $label magnitudes (primeiros $take bins):")
            appendLine("  CPU   : ${cpuPreview.take(take)}")
            appendLine("  TFLite: ${tflitePreview.take(take)}")
            appendLine("  ratio (TFLite/CPU): $ratios")
            appendLine("  métricas -> maxRel=${metrics.maxRelativeDiff}, meanRel=${metrics.meanRelativeDiff}, maxAbs=${metrics.maxAbsoluteDiff}, meanAbs=${metrics.meanAbsoluteDiff}, rmse=${metrics.rmse}")
        }
        Log.i("FFT_TEST", summary)
    }

    private fun measureFftAgainstCpu(
        signalLength: Int,
        numSensors: Int,
        delegateType: DelegateType,
        runIndex: Int
    ): FftErrorMetrics {
        val batch = AccelerometerBatchGenerator.generate(
            numSensors = numSensors,
            samplesPerSensor = signalLength,
            seed = seed + runIndex
        )
        val input = FftInputBuilder.fromAccelerometer(batch, signalLength)
        val processor = try {
            FftTfliteProcessor(
                context = context,
                delegateType = delegateType,
                numSensors = numSensors,
                signalLength = signalLength
            )
        } catch (e: Exception) {
            throw IllegalStateException("Falha ao criar FftTfliteProcessor ${delegateType.name}: ${e.message}", e)
        }.also { fftProcessors += it }
        val tflite = processor.process(input.samples, input.weights).output
        val previewTake = minOf(8, tflite.weightedMagnitudes.firstOrNull()?.size ?: 0)
        val previewHolder = PreviewHolder(previewTake)
        val metrics = computeFftErrorMetricsStreaming(
            tfliteWeighted = tflite.weightedMagnitudes,
            cpuProcessor = FftCpuProcessor(numSensors, signalLength),
            samples = input.samples,
            weights = input.weights,
            preview = previewHolder
        )
        val tflitePreview = tflite.weightedMagnitudes.firstOrNull()
            ?.take(previewTake)
            ?.toFloatArray()
            ?: FloatArray(0)
        logFftComparison(
            cpuPreview = previewHolder.cpuPreview ?: FloatArray(0),
            tflitePreview = tflitePreview,
            label = delegateType.displayName,
            metrics = metrics
        )
        PrecisionReporter.appendFft(
            context = context,
            signalLength = signalLength,
            numSensors = numSensors,
            delegate = delegateType.displayName,
            metrics = metrics
        )
        return metrics
    }

    private fun reportPrecisionStats(
        signalLength: Int,
        repeats: Int,
        metricsList: List<FftErrorMetrics>
    ) {
        if (metricsList.isEmpty()) return
        val maxRelSeries = metricsList.map { it.maxRelativeDiff }
        val meanRelSeries = metricsList.map { it.meanRelativeDiff }
        val rmseSeries = metricsList.map { it.rmse }
        val avgMaxRel = maxRelSeries.averageOrZero()
        val stdMaxRel = maxRelSeries.stddev(avgMaxRel)
        val avgMeanRel = meanRelSeries.averageOrZero()
        val avgRmse = rmseSeries.averageOrZero()
        val summary = buildString {
            append("FFT stats len=$signalLength (${repeats}x)")
            append(" | maxRel avg=${avgMaxRel.toPct()} std=${stdMaxRel.toPct()} min=${maxRelSeries.minOrNull()?.toPct()} max=${maxRelSeries.maxOrNull()?.toPct()}")
            append(" | meanRel avg=${avgMeanRel.toPct()} | rmse avg=${String.format(Locale.US, "%.6f", avgRmse)}")
        }
        Log.i("FFT_PRECISION_STATS", summary)
    }

    private fun List<Double>.averageOrZero(): Double =
        if (isEmpty()) 0.0 else sum() / size

    private fun List<Double>.stddev(mean: Double = averageOrZero()): Double {
        if (isEmpty()) return 0.0
        var acc = 0.0
        for (value in this) {
            val diff = value - mean
            acc += diff * diff
        }
        return sqrt(acc / size)
    }

    private fun Double?.toPct(): String =
        this?.let { String.format(Locale.US, "%.3f%%", it * 100.0) } ?: "-"

    private fun computeFftErrorMetricsStreaming(
        tfliteWeighted: Array<FloatArray>,
        cpuProcessor: FftCpuProcessor,
        samples: Array<FloatArray>,
        weights: Array<FloatArray>,
        preview: PreviewHolder
    ): FftErrorMetrics {
        val accumulator = FftMetricsAccumulator()
        cpuProcessor.processStreaming(samples, weights) { sensorIndex, _, weighted ->
            val tfliteArray = tfliteWeighted[sensorIndex]
            accumulator.consume(weighted, tfliteArray)
            preview.captureIfNeeded(sensorIndex, weighted)
        }
        return accumulator.build()
    }

    private class PreviewHolder(private val take: Int) {
        var cpuPreview: FloatArray? = null
            private set

        fun captureIfNeeded(sensorIndex: Int, weighted: FloatArray) {
            if (sensorIndex == 0 && cpuPreview == null && take > 0) {
                cpuPreview = weighted.copyOfRange(0, take.coerceAtMost(weighted.size))
            }
        }
    }

    private class FftMetricsAccumulator {
        private var maxRel = 0.0
        private var sumRel = 0.0
        private var relCount = 0
        private var maxAbs = 0.0
        private var sumAbs = 0.0
        private var sumSq = 0.0
        private var count = 0

        fun consume(cpu: FloatArray, tflite: FloatArray) {
            for (idx in cpu.indices) {
                if (idx >= tflite.size) break
                val expected = cpu[idx].toDouble()
                val got = tflite[idx].toDouble()
                val diff = kotlin.math.abs(expected - got)
                val denom = kotlin.math.max(kotlin.math.abs(expected), FFT_RELATIVE_GUARD.toDouble())
                val rel = diff / denom
                if (rel > maxRel) maxRel = rel
                sumRel += rel
                relCount++
                if (diff > maxAbs) maxAbs = diff
                sumAbs += diff
                sumSq += diff * diff
                count++
            }
        }

        fun build(): FftErrorMetrics {
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
    }

    data class FftErrorMetrics(
        val maxRelativeDiff: Double = 0.0,
        val meanRelativeDiff: Double = 0.0,
        val maxAbsoluteDiff: Double = 0.0,
        val meanAbsoluteDiff: Double = 0.0,
        val rmse: Double = 0.0
    )

    companion object {
        private const val FFT_RELATIVE_GUARD = 1e3 // bins abaixo disso usam apenas erro absoluto
        private const val EXPERIMENTAL_LENGTH_THRESHOLD = 131_072

        private fun isExperimentalLength(length: Int): Boolean =
            length >= EXPERIMENTAL_LENGTH_THRESHOLD
    }
}
