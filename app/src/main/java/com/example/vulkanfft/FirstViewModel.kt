package com.example.vulkanfft

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.vulkanfft.logging.BenchmarkEntry
import com.example.vulkanfft.logging.BenchmarkReporter
import com.example.vulkanfft.logging.DeviceInfoProvider
import com.example.vulkanfft.logging.EnergyLogEntry
import com.example.vulkanfft.logging.EnergyReporter
import com.example.vulkanfft.logging.ResultLogger
import com.example.vulkanfft.util.AccelerometerBatchGenerator
import com.example.vulkanfft.util.DelegateType
import com.example.vulkanfft.util.FftCpuProcessor
import com.example.vulkanfft.util.FftInputBuilder
import com.example.vulkanfft.util.FftResult
import com.example.vulkanfft.util.FftTfliteProcessor
import com.example.vulkanfft.util.InferenceTiming
import com.example.vulkanfft.util.StatModelProcessor
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

class FirstViewModel : ViewModel() {

    private val _benchmarkResult = MutableLiveData("Aguardando execuções.")
    val benchmarkResult: LiveData<String> = _benchmarkResult

    private val _energyResult = MutableLiveData("Sem testes energéticos recentes.")
    val energyResult: LiveData<String> = _energyResult

    private val _progress = MutableLiveData<BenchmarkProgress>()
    val progress: LiveData<BenchmarkProgress> = _progress

    private val _energyInProgress = MutableLiveData(false)
    val energyInProgress: LiveData<Boolean> = _energyInProgress

    private var accelerometerBatch: AccelerometerBatchGenerator.AccelerometerBatch? = null
    private var madInput: Array<IntArray>? = null
    private var fftInput: FftInputBuilder.FftProcessorInput? = null
    private val fftCpuProcessor = FftCpuProcessor(
        numSensors = FFT_NUM_SENSORS,
        signalLength = FFT_SIGNAL_LENGTH
    )
    private var energyTestJob: Job? = null

    fun runScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int = DEFAULT_BENCH_REPETITIONS
    ) {
        viewModelScope.launch(Dispatchers.Default) {
            val summary = executeScenario(context.applicationContext, scenario, iterations)
            _benchmarkResult.postValue(summary.summaryText)
        }
    }

    fun startEnergyTest(context: Context) {
        if (energyTestJob?.isActive == true) {
            _energyResult.postValue("Teste energético já está em execução.")
            return
        }
        val ctx = context.applicationContext
        energyTestJob = viewModelScope.launch(Dispatchers.Default) {
            _energyInProgress.postValue(true)
            val scenarioSummaries = mutableListOf<String>()
            try {
                ENERGY_TEST_SCENARIOS.forEach { scenario ->
                    ensureActive()
                    val startSnapshot = captureBatterySnapshot(ctx)
                    val scenarioStart = System.currentTimeMillis()
                    repeat(ENERGY_TEST_ITERATIONS) {
                        ensureActive()
                        executeScenario(
                            context = ctx,
                            scenario = scenario,
                            iterations = 1,
                            logResults = false
                        )
                    }
                    val elapsed = System.currentTimeMillis() - scenarioStart
                    val endSnapshot = captureBatterySnapshot(ctx)
                    val dropPercent = startSnapshot.levelPercent - endSnapshot.levelPercent
                    scenarioSummaries += "${scenario.displayLabel}: Δ ${"%.2f".format(dropPercent)}% | Tempo ${formatDuration(elapsed)}"
                    ResultLogger.append(
                        ctx,
                        "energy_tests.txt",
                        listOf(
                            "Cenário energético: ${scenario.displayLabel}",
                            "Execuções: $ENERGY_TEST_ITERATIONS",
                            "Tempo: ${formatDuration(elapsed)}",
                            "Bateria: ${"%.2f".format(startSnapshot.levelPercent)}% -> ${"%.2f".format(endSnapshot.levelPercent)}% (Δ ${"%.2f".format(dropPercent)}%)"
                        )
                    )
                    EnergyReporter.append(
                        ctx,
                        EnergyLogEntry(
                            timestamp = System.currentTimeMillis(),
                            scenarioLabel = scenario.displayLabel,
                            delegate = scenario.delegateMode.displayName,
                            batchMode = scenario.batchMode.displayName,
                            durationMinutes = maxOf(1, (elapsed / 60000.0).roundToInt()),
                            totalRuns = ENERGY_TEST_ITERATIONS,
                            intervalSeconds = 0,
                            batteryStartPercent = startSnapshot.levelPercent,
                            batteryEndPercent = endSnapshot.levelPercent,
                            batteryDropPercent = dropPercent,
                            batteryStartChargeMah = startSnapshot.chargeMah,
                            batteryEndChargeMah = endSnapshot.chargeMah,
                            energyStartNwh = startSnapshot.energyNwh,
                            energyEndNwh = endSnapshot.energyNwh,
                            temperatureStartC = startSnapshot.temperatureC,
                            temperatureEndC = endSnapshot.temperatureC,
                            isChargingStart = startSnapshot.isCharging,
                            isChargingEnd = endSnapshot.isCharging,
                            powerSaveStart = startSnapshot.powerSave,
                            powerSaveEnd = endSnapshot.powerSave,
                            notes = "Tempo ${formatDuration(elapsed)} | Δ ${"%.2f".format(dropPercent)}%"
                        )
                    )
                }
                val finalSummary = buildString {
                    appendLine("Teste energético finalizado.")
                    scenarioSummaries.forEach { appendLine(it) }
                }
                _energyResult.postValue(finalSummary.trim())
            } catch (ex: CancellationException) {
                _energyResult.postValue("Teste energético cancelado pelo usuário.")
                throw ex
            } finally {
                _energyInProgress.postValue(false)
            }
        }.also { job ->
            job.invokeOnCompletion { energyTestJob = null }
        }
    }

    fun cancelEnergyTest() {
        energyTestJob?.cancel()
    }

    fun runFullBenchmarkSuite(context: Context) {
        viewModelScope.launch(Dispatchers.Default) {
            val ctx = context.applicationContext
            val tasks = BenchmarkScenario.suiteOrder
            _progress.postValue(
                BenchmarkProgress(
                    total = tasks.size,
                    current = 0,
                    running = true,
                    message = "Iniciando suíte de testes"
                )
            )
            tasks.forEachIndexed { index, scenario ->
                val summary = executeScenario(ctx, scenario, DEFAULT_BENCH_REPETITIONS)
                _benchmarkResult.postValue(summary.summaryText)
                _progress.postValue(
                    BenchmarkProgress(
                        total = tasks.size,
                        current = index + 1,
                        running = index + 1 < tasks.size,
                        message = "Executado: ${scenario.displayLabel}"
                    )
                )
            }
            _progress.postValue(
                BenchmarkProgress(
                    total = tasks.size,
                    current = tasks.size,
                    running = false,
                    message = "Suíte concluída"
                )
            )
        }
    }

    private suspend fun executeScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        logResults: Boolean = true
    ): ScenarioRunSummary {
        return when (scenario.algorithm) {
            Algorithm.MAD -> runMadScenario(context, scenario, iterations, logResults)
            Algorithm.FFT -> runFftScenario(context, scenario, iterations, logResults)
        }
    }

    private suspend fun runMadScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        logResults: Boolean
    ): ScenarioRunSummary {
        val inputs = buildMadInputs(scenario.batchMode)
        val readings = inputs.map { buildMadReadings(it) }
        val timingSamples = mutableListOf<InferenceTiming>()
        var lastResult = "n/d"

        if (scenario.delegateMode == DelegateMode.CPU_NATIVE) {
            repeat(iterations) {
                val start = System.nanoTime()
                readings.forEach { reading ->
                    val result = getMAD(reading)
                    lastResult = formatMadResult(result)
                }
                val computeMs = nanosToMillis(System.nanoTime() - start)
                timingSamples += InferenceTiming(transferMs = 0.0, computeMs = computeMs)
            }
        } else {
            val delegateType = scenario.delegateMode.delegateType
                ?: error("DelegateType required for ${scenario.displayLabel}")
            val processor = StatModelProcessor(
                context = context,
                delegateType = delegateType,
                sizeVector = MAD_VECTOR_LENGTH
            )
            repeat(iterations) {
                var transfer = 0.0
                var compute = 0.0
                var lastOutput: FloatArray? = null
                inputs.forEach { input ->
                    val result = processor.process(input)
                    transfer += result.timing.transferMs
                    compute += result.timing.computeMs
                    lastOutput = result.output
                }
                lastResult = formatTfliteMadOutput(lastOutput)
                timingSamples += InferenceTiming(transferMs = transfer, computeMs = compute)
            }
            processor.close()
        }

        val batchSize = inputs.size
        val timingStats = buildTimingStats(timingSamples)
        val throughput = computeThroughput(
            operations = MAD_VECTOR_LENGTH.toDouble() * batchSize,
            durationMs = timingStats.total.mean
        )
        val dataDescription = "${batchSize}×(1 sensor × $MAD_VECTOR_LENGTH amostras)"
        val summaryText = buildString {
            appendLine("${scenario.displayLabel} (${scenario.batchMode.displayName})")
            appendLine("Total: ${formatStats(timingStats.total)} | Transfer: ${formatStats(timingStats.transfer)} | Proc: ${formatStats(timingStats.compute)}")
            appendLine("Último resultado: $lastResult")
        }

        if (logResults) {
            logScenario(
                context = context,
                scenario = scenario,
                iterations = iterations,
                batchSize = batchSize,
                dataDescription = dataDescription,
                inputSize = MAD_VECTOR_LENGTH * batchSize,
                throughput = throughput,
                timingStats = timingStats,
                notes = "$lastResult | Tempos: ${formatTimingSamples(timingSamples)}"
            )
        }

        return ScenarioRunSummary(
            scenario = scenario,
            summaryText = summaryText,
            timings = timingStats,
            notes = lastResult,
            batchSize = batchSize
        )
    }

    private suspend fun runFftScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        logResults: Boolean
    ): ScenarioRunSummary {
        val inputs = buildFftInputs(scenario.batchMode)
        val timingSamples = mutableListOf<InferenceTiming>()
        var lastSummary = "-"

        if (scenario.delegateMode == DelegateMode.CPU_NATIVE) {
            repeat(iterations) {
                var compute = 0.0
                inputs.forEach { input ->
                    val start = System.nanoTime()
                    val result = fftCpuProcessor.process(input.samples, input.weights)
                    val durationMs = nanosToMillis(System.nanoTime() - start)
                    compute += durationMs
                    lastSummary = formatFftSummary(
                        label = "FFT CPU",
                        durationMs = durationMs,
                        result = result
                    )
                }
                timingSamples += InferenceTiming(transferMs = 0.0, computeMs = compute)
            }
        } else {
            val delegateType = scenario.delegateMode.delegateType
                ?: error("DelegateType required for ${scenario.displayLabel}")
            val processor = FftTfliteProcessor(
                context = context,
                delegateType = delegateType,
                numSensors = FFT_NUM_SENSORS,
                signalLength = FFT_SIGNAL_LENGTH
            )
            repeat(iterations) {
                var transfer = 0.0
                var compute = 0.0
                inputs.forEach { input ->
                    val result = processor.process(input.samples, input.weights)
                    transfer += result.timing.transferMs
                    compute += result.timing.computeMs
                    lastSummary = formatFftSummary(
                        label = "FFT TFLite (${scenario.delegateMode.displayName})",
                        durationMs = result.timing.totalMs,
                        result = result.output
                    )
                }
                timingSamples += InferenceTiming(transferMs = transfer, computeMs = compute)
            }
            processor.close()
        }

        val batchSize = inputs.size
        val timingStats = buildTimingStats(timingSamples)
        val operations = FFT_SIGNAL_LENGTH.toDouble() * FFT_NUM_SENSORS * batchSize
        val throughput = computeThroughput(operations, timingStats.total.mean)
        val dataDescription = "${batchSize}×($FFT_NUM_SENSORS sensores × $FFT_SIGNAL_LENGTH amostras)"
        val summaryText = buildString {
            appendLine("${scenario.displayLabel} (${scenario.batchMode.displayName})")
            appendLine("Total: ${formatStats(timingStats.total)} | Transfer: ${formatStats(timingStats.transfer)} | Proc: ${formatStats(timingStats.compute)}")
            appendLine("Highlights: $lastSummary")
        }

        if (logResults) {
            logScenario(
                context = context,
                scenario = scenario,
                iterations = iterations,
                batchSize = batchSize,
                dataDescription = dataDescription,
                inputSize = FFT_NUM_SENSORS * FFT_SIGNAL_LENGTH * batchSize,
                throughput = throughput,
                timingStats = timingStats,
                notes = "$lastSummary | Tempos: ${formatTimingSamples(timingSamples)}"
            )
        }

        return ScenarioRunSummary(
            scenario = scenario,
            summaryText = summaryText,
            timings = timingStats,
            notes = lastSummary,
            batchSize = batchSize
        )
    }

    private fun logScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        batchSize: Int,
        dataDescription: String,
        inputSize: Int,
        throughput: Double,
        timingStats: TimingStats,
        notes: String
    ) {
        ResultLogger.append(
            context,
            scenario.logFileName,
            listOf(
                "Cenário: ${scenario.displayLabel}",
                "Execuções: $iterations | Pacotes: $batchSize",
                "Total(ms): ${formatStats(timingStats.total)}",
                "Transfer(ms): ${formatStats(timingStats.transfer)}",
                "Processamento(ms): ${formatStats(timingStats.compute)}",
                "Notas: $notes"
            )
        )

        val deviceInfo = DeviceInfoProvider.collect(context)
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = scenario.displayLabel,
                processingMode = scenario.processingMode,
                delegate = scenario.delegateMode.displayName,
                dataDescription = dataDescription,
                inputSize = inputSize,
                durationMs = timingStats.total.mean,
                durationStdDev = timingStats.total.stdDev,
                durationMin = timingStats.total.min,
                durationMax = timingStats.total.max,
                transferDurationMs = timingStats.transfer.mean,
                transferStdDev = timingStats.transfer.stdDev,
                transferMin = timingStats.transfer.min,
                transferMax = timingStats.transfer.max,
                computeDurationMs = timingStats.compute.mean,
                computeStdDev = timingStats.compute.stdDev,
                computeMin = timingStats.compute.min,
                computeMax = timingStats.compute.max,
                throughput = throughput,
                iterations = iterations,
                batchSize = batchSize,
                estimatedEnergyImpact = describeEnergy(scenario.delegateMode),
                deviceInfo = deviceInfo,
                extraNotes = notes
            )
        )
    }

    private fun buildMadInputs(batchMode: BatchMode): List<Array<IntArray>> {
        return when (batchMode) {
            BatchMode.SINGLE -> listOf(ensureMadInput())
            BatchMode.TEN_PACKETS -> {
                val batch = ensureAccelerometerBatch()
                batch.sensors.take(BATCH_SIZE).map { it.asMadInput() }
            }
        }
    }

    private fun buildFftInputs(batchMode: BatchMode): List<FftInputBuilder.FftProcessorInput> {
        val input = ensureFftInput()
        return when (batchMode) {
            BatchMode.SINGLE -> listOf(input)
            BatchMode.TEN_PACKETS -> List(BATCH_SIZE) { input }
        }
    }

    private data class BatterySnapshot(
        val levelPercent: Double,
        val isCharging: Boolean,
        val temperatureC: Double?,
        val energyNwh: Double?,
        val chargeMah: Double?,
        val powerSave: Boolean
    )

    private fun captureBatterySnapshot(context: Context): BatterySnapshot {
        val bm = context.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val intent = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val level = intent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = intent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val percent = if (level >= 0 && scale > 0) level * 100.0 / scale else Double.NaN
        val plugged = intent?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0
        val isCharging = plugged != 0
        val temp = intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        val temperatureC = temp?.takeIf { it != Int.MIN_VALUE }?.div(10.0)
        val energyCounter = bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER)
            .takeIf { it != Long.MIN_VALUE }
            ?.div(1_000_000_000.0)
        val chargeCounter = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER)
            .takeIf { it != Int.MIN_VALUE }
            ?.div(1000.0)

        return BatterySnapshot(
            levelPercent = percent,
            isCharging = isCharging,
            temperatureC = temperatureC,
            energyNwh = energyCounter,
            chargeMah = chargeCounter,
            powerSave = pm.isPowerSaveMode
        )
    }

    private fun buildTimingStats(samples: List<InferenceTiming>): TimingStats {
        val totals = samples.map { it.totalMs }
        val transfers = samples.map { it.transferMs }
        val computes = samples.map { it.computeMs }
        return TimingStats(
            total = computeStats(totals),
            transfer = computeStats(transfers),
            compute = computeStats(computes)
        )
    }

    private fun computeStats(values: List<Double>): StatsSummary {
        if (values.isEmpty()) return StatsSummary.ZERO
        val mean = values.average()
        val variance = values.map { (it - mean).pow(2) }.average()
        return StatsSummary(
            mean = mean,
            stdDev = sqrt(variance),
            min = values.minOrNull() ?: mean,
            max = values.maxOrNull() ?: mean
        )
    }

    private fun formatStats(stats: StatsSummary): String {
        return "${"%.3f".format(stats.mean)}±${"%.3f".format(stats.stdDev)} (min=${"%.3f".format(stats.min)} | max=${"%.3f".format(stats.max)})"
    }

    private fun formatTimingSamples(samples: List<InferenceTiming>): String {
        return samples.joinToString { sample ->
            "T=${"%.2f".format(sample.transferMs)}ms/P=${"%.2f".format(sample.computeMs)}ms"
        }
    }

    private fun describeEnergy(delegateMode: DelegateMode): String {
        return when (delegateMode) {
            DelegateMode.CPU_NATIVE -> "Alta (CPU Kotlin)"
            DelegateMode.TFLITE_CPU -> "Média (CPU TFLite)"
            DelegateMode.TFLITE_GPU -> "Baixa (GPU Paralela)"
            DelegateMode.TFLITE_NNAPI -> "Baixa/Média (NNAPI)"
        }
    }

    private fun formatMadResult(result: MADResult): String {
        return "Mean=${"%.3f".format(result.mean)} | Std=${"%.3f".format(result.stdDev)} | Min=${"%.3f".format(result.min)} | Max=${"%.3f".format(result.max)}"
    }

    private fun formatTfliteMadOutput(output: FloatArray?): String {
        if (output == null) return "-"
        return "Mean=${output.getOrNull(0)} | Std=${output.getOrNull(1)} | Min=${output.getOrNull(2)} | Max=${output.getOrNull(3)}"
    }

    data class MADResult(
        val mean: Double,
        val stdDev: Double,
        val min: Double,
        val max: Double
    )

    private fun getMAD(readings: List<AccelerometerSample>): MADResult {
        val magnitudeByTime = readings.map {
            val magnitude = sqrt((it.x * it.x + it.y * it.y + it.z * it.z).toDouble())
            Pair(it.timestamp, magnitude)
        }
        val windowSizeMillis = TimeUnit.SECONDS.toMillis(5)
        val startTime = magnitudeByTime.first().first
        val blockAverages = mutableListOf<Double>()
        var currentWindowStart = startTime

        while (true) {
            val currentWindowEnd = currentWindowStart + windowSizeMillis
            val windowSamples = magnitudeByTime.filter { it.first in currentWindowStart until currentWindowEnd }
                .map { it.second }
            if (windowSamples.isEmpty()) break
            blockAverages.add(windowSamples.average())
            currentWindowStart = currentWindowEnd
            if (currentWindowStart > magnitudeByTime.last().first) break
        }

        val mean = blockAverages.average()
        val variance = blockAverages.sumOf { (it - mean) * (it - mean) } / blockAverages.size
        return MADResult(
            mean = mean,
            stdDev = sqrt(variance),
            min = blockAverages.minOrNull() ?: 0.0,
            max = blockAverages.maxOrNull() ?: 0.0
        )
    }

    data class AccelerometerSample(
        val timestamp: Long,
        val x: Int,
        val y: Int,
        val z: Int
    )

    data class BenchmarkProgress(
        val total: Int,
        val current: Int,
        val running: Boolean,
        val message: String
    )

    private fun buildMadReadings(input: Array<IntArray>): List<AccelerometerSample> {
        val timestamps = input[0]
        val x = input[1]
        val y = input[2]
        val z = input[3]
        return timestamps.indices.map { i ->
            AccelerometerSample(
                timestamp = timestamps[i].toLong(),
                x = x[i],
                y = y[i],
                z = z[i]
            )
        }
    }

    private fun formatFftSummary(
        label: String,
        durationMs: Double,
        result: FftResult
    ): String {
        val totals = result.weightedMagnitudes.mapIndexed { index, weights ->
            "S$index=${"%.2f".format(weights.sum())}"
        }.take(4)
        val snippet = totals.joinToString(" | ")
        val firstSensorBins = result.weightedMagnitudes.firstOrNull()
            ?.take(3)
            ?.joinToString(", ") { "%.2f".format(it) }
            ?: "-"

        return buildString {
            appendLine("$label -> ${"%.3f".format(durationMs)} ms")
            appendLine("Σ pesos (4 primeiros): $snippet")
            append("S0 bins[0..2]: $firstSensorBins")
        }
    }

    private fun computeThroughput(operations: Double, durationMs: Double): Double {
        if (durationMs <= 0.0) return 0.0
        return operations / (durationMs / 1000.0)
    }

    private fun ensureAccelerometerBatch(): AccelerometerBatchGenerator.AccelerometerBatch {
        if (accelerometerBatch == null) {
            accelerometerBatch = AccelerometerBatchGenerator.generate(
                numSensors = FFT_NUM_SENSORS,
                samplesPerSensor = MAD_VECTOR_LENGTH
            )
            madInput = null
            fftInput = null
        }
        return accelerometerBatch!!
    }

    private fun ensureMadInput(): Array<IntArray> {
        if (madInput == null) {
            val batch = ensureAccelerometerBatch()
            madInput = batch.sensors.first().asMadInput()
        }
        return madInput!!
    }

    private fun ensureFftInput(): FftInputBuilder.FftProcessorInput {
        if (fftInput == null) {
            val batch = ensureAccelerometerBatch()
            fftInput = FftInputBuilder.fromAccelerometer(batch, FFT_SIGNAL_LENGTH)
        }
        return fftInput!!
    }

    private fun nanosToMillis(value: Long): Double = value / 1_000_000.0

    private fun formatDuration(durationMs: Long): String {
        val minutes = TimeUnit.MILLISECONDS.toMinutes(durationMs)
        val seconds =
            TimeUnit.MILLISECONDS.toSeconds(durationMs) - TimeUnit.MINUTES.toSeconds(minutes)
        val millis =
            durationMs - TimeUnit.SECONDS.toMillis(TimeUnit.MILLISECONDS.toSeconds(durationMs))
        return String.format("%02d:%02d.%03d", minutes, seconds, millis)
    }

    companion object {
        private const val FFT_NUM_SENSORS = 10
        private const val FFT_SIGNAL_LENGTH = 4096
        private const val MAD_VECTOR_LENGTH = 4096
        private const val DEFAULT_BENCH_REPETITIONS = 10
        private const val BATCH_SIZE = 10
        private const val ENERGY_TEST_ITERATIONS = 100
        private val ENERGY_TEST_SCENARIOS = listOf(
            BenchmarkScenario.MAD_CPU_SINGLE,
            BenchmarkScenario.MAD_TFLITE_CPU_SINGLE,
            BenchmarkScenario.MAD_TFLITE_GPU_SINGLE,
            BenchmarkScenario.MAD_TFLITE_NN_SINGLE
        )
    }
}

data class ScenarioRunSummary(
    val scenario: BenchmarkScenario,
    val summaryText: String,
    val timings: TimingStats,
    val notes: String,
    val batchSize: Int
)

data class StatsSummary(
    val mean: Double,
    val stdDev: Double,
    val min: Double,
    val max: Double
) {
    companion object {
        val ZERO = StatsSummary(0.0, 0.0, 0.0, 0.0)
    }
}

data class TimingStats(
    val total: StatsSummary,
    val transfer: StatsSummary,
    val compute: StatsSummary
)

enum class Algorithm { MAD, FFT }

enum class BatchMode(val displayName: String) {
    SINGLE("1 pacote"),
    TEN_PACKETS("10 pacotes")
}

enum class DelegateMode(
    val displayName: String,
    val delegateType: DelegateType?
) {
    CPU_NATIVE("CPU Kotlin", null),
    TFLITE_CPU("TFLite CPU", DelegateType.CPU),
    TFLITE_GPU("TFLite GPU", DelegateType.GPU),
    TFLITE_NNAPI("TFLite NNAPI", DelegateType.NNAPI)
}

enum class BenchmarkScenario(
    val displayLabel: String,
    val algorithm: Algorithm,
    val delegateMode: DelegateMode,
    val batchMode: BatchMode,
    val logFileName: String,
    val processingMode: String
) {
    MAD_CPU_SINGLE(
        "MAD CPU Kotlin",
        Algorithm.MAD,
        DelegateMode.CPU_NATIVE,
        BatchMode.SINGLE,
        "mad_cpu_single.txt",
        "Kotlin"
    ),
    MAD_TFLITE_CPU_SINGLE(
        "MAD TFLite CPU",
        Algorithm.MAD,
        DelegateMode.TFLITE_CPU,
        BatchMode.SINGLE,
        "mad_tflite_cpu_single.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_GPU_SINGLE(
        "MAD TFLite GPU",
        Algorithm.MAD,
        DelegateMode.TFLITE_GPU,
        BatchMode.SINGLE,
        "mad_tflite_gpu_single.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_NN_SINGLE(
        "MAD TFLite NNAPI",
        Algorithm.MAD,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.SINGLE,
        "mad_tflite_nn_single.txt",
        "TensorFlow Lite"
    ),
    MAD_CPU_BATCH(
        "MAD CPU Kotlin x10",
        Algorithm.MAD,
        DelegateMode.CPU_NATIVE,
        BatchMode.TEN_PACKETS,
        "mad_cpu_batch.txt",
        "Kotlin"
    ),
    MAD_TFLITE_CPU_BATCH(
        "MAD TFLite CPU x10",
        Algorithm.MAD,
        DelegateMode.TFLITE_CPU,
        BatchMode.TEN_PACKETS,
        "mad_tflite_cpu_batch.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_GPU_BATCH(
        "MAD TFLite GPU x10",
        Algorithm.MAD,
        DelegateMode.TFLITE_GPU,
        BatchMode.TEN_PACKETS,
        "mad_tflite_gpu_batch.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_NN_BATCH(
        "MAD TFLite NNAPI x10",
        Algorithm.MAD,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.TEN_PACKETS,
        "mad_tflite_nn_batch.txt",
        "TensorFlow Lite"
    ),
    FFT_CPU_SINGLE(
        "FFT CPU",
        Algorithm.FFT,
        DelegateMode.CPU_NATIVE,
        BatchMode.SINGLE,
        "fft_cpu_single.txt",
        "Kotlin"
    ),
    FFT_TFLITE_CPU_SINGLE(
        "FFT TFLite CPU",
        Algorithm.FFT,
        DelegateMode.TFLITE_CPU,
        BatchMode.SINGLE,
        "fft_tflite_cpu_single.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_GPU_SINGLE(
        "FFT TFLite GPU",
        Algorithm.FFT,
        DelegateMode.TFLITE_GPU,
        BatchMode.SINGLE,
        "fft_tflite_gpu_single.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_NN_SINGLE(
        "FFT TFLite NNAPI",
        Algorithm.FFT,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.SINGLE,
        "fft_tflite_nn_single.txt",
        "TensorFlow Lite"
    ),
    FFT_CPU_BATCH(
        "FFT CPU x10",
        Algorithm.FFT,
        DelegateMode.CPU_NATIVE,
        BatchMode.TEN_PACKETS,
        "fft_cpu_batch.txt",
        "Kotlin"
    ),
    FFT_TFLITE_CPU_BATCH(
        "FFT TFLite CPU x10",
        Algorithm.FFT,
        DelegateMode.TFLITE_CPU,
        BatchMode.TEN_PACKETS,
        "fft_tflite_cpu_batch.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_GPU_BATCH(
        "FFT TFLite GPU x10",
        Algorithm.FFT,
        DelegateMode.TFLITE_GPU,
        BatchMode.TEN_PACKETS,
        "fft_tflite_gpu_batch.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_NN_BATCH(
        "FFT TFLite NNAPI x10",
        Algorithm.FFT,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.TEN_PACKETS,
        "fft_tflite_nn_batch.txt",
        "TensorFlow Lite"
    );

    companion object {
        val suiteOrder = values().toList()
    }
}
