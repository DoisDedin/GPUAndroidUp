package com.example.vulkanfft

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.PowerManager
import android.util.Log
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
import kotlinx.coroutines.ensureActive
import java.util.concurrent.TimeUnit
import kotlin.coroutines.coroutineContext
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sqrt

class BenchmarkExecutor {

    private val accelerometerBatches =
        mutableMapOf<Int, AccelerometerBatchGenerator.AccelerometerBatch>()
    private val madInputsCache = mutableMapOf<Int, Array<IntArray>>()
    private val fftInputsCache =
        mutableMapOf<Int, FftInputBuilder.FftProcessorInput>()
    private val fftCpuProcessors = mutableMapOf<Int, FftCpuProcessor>()

    suspend fun runScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        scale: DataScale = DataScale.BASE,
        logResults: Boolean = true
    ): ScenarioRunSummary {
        return executeScenario(context, scenario, iterations, scale, logResults)
    }

    suspend fun runEnergyTests(
        context: Context,
        iterations: Int = ENERGY_TEST_ITERATIONS,
        scenarios: List<BenchmarkScenario> = BenchmarkScenario.suiteOrder,
        madScale: DataScale = DataScale.BASE,
        fftScale: DataScale = DataScale.BASE,
        onScenarioStart: (current: Int, total: Int, label: String, scale: DataScale) -> Unit = { _, _, _, _ -> },
        onScenarioComplete: (EnergyScenarioSnapshot) -> Unit = {}
    ): String {
        val ctx = context.applicationContext
        val scenarioSummaries = mutableListOf<String>()
        val totalScenarios = scenarios.size
        scenarios.forEachIndexed { index, scenario ->
            val scenarioNumber = index + 1
            val scale = when (scenario.algorithm) {
                Algorithm.MAD -> madScale
                Algorithm.FFT -> fftScale
            }
            onScenarioStart(scenarioNumber, totalScenarios, scenario.displayLabel, scale)
            coroutineContext.ensureActive()
            val startSnapshot = captureBatterySnapshot(ctx)
            val scenarioStart = System.currentTimeMillis()
            val runSummary = executeScenario(
                context = ctx,
                scenario = scenario,
                iterations = iterations,
                scale = scale,
                logResults = false
            )
            val elapsed = System.currentTimeMillis() - scenarioStart
            val endSnapshot = captureBatterySnapshot(ctx)
            val dropPercent = startSnapshot.levelPercent - endSnapshot.levelPercent
                    val totalStatsText = formatStats(runSummary.timings.total)
                    val tempSummary = formatTemperatureRange(startSnapshot.temperatureC, endSnapshot.temperatureC)
                    val dataLength = formatDataLength(scenario.algorithm, scale)
                    val scenarioLine = buildString {
                        append("${scenario.displayLabel} (${scale.shortLabel})")
                        append(": Δ ${"%.2f".format(dropPercent)}% | Tempo ${formatDuration(elapsed)}")
                        append(" | Total $totalStatsText")
                        append(" | Vetor $dataLength")
                        tempSummary?.let { append(" | $it") }
                    }
                    scenarioSummaries += scenarioLine
                    val percentComplete =
                        ((scenarioNumber.toDouble() / totalScenarios) * 100).roundToInt().coerceIn(0, 100)
                    val energySnapshot = EnergyScenarioSnapshot(
                scenario = scenario,
                summaryLine = scenarioLine,
                elapsedMs = elapsed,
                dropPercent = dropPercent,
                        totalStats = runSummary.timings.total,
                        scenarioIndex = scenarioNumber,
                        totalScenarios = totalScenarios,
                        percentComplete = percentComplete,
                        scale = scale,
                        temperatureStartC = startSnapshot.temperatureC,
                        temperatureEndC = endSnapshot.temperatureC
                    )
                    onScenarioComplete(energySnapshot)
                    logEnergyResult(
                        context = ctx,
                scenario = scenario,
                iterations = iterations,
                elapsed = elapsed,
                startSnapshot = startSnapshot,
                endSnapshot = endSnapshot,
                dropPercent = dropPercent,
                runSummary = runSummary,
                        totalStatsText = totalStatsText,
                        scale = scale,
                        dataLength = dataLength,
                        temperatureSummary = tempSummary
                    )
                }
        val finalSummary = buildString {
            appendLine("Teste energético finalizado.")
            scenarioSummaries.forEach { appendLine(it) }
        }
        return finalSummary.trim()
    }

    private suspend fun executeScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        scale: DataScale,
        logResults: Boolean = true
    ): ScenarioRunSummary {
        return when (scenario.algorithm) {
            Algorithm.MAD -> runMadScenario(context, scenario, iterations, scale, logResults)
            Algorithm.FFT -> runFftScenario(context, scenario, iterations, scale, logResults)
        }
    }

    private suspend fun runMadScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        scale: DataScale,
        logResults: Boolean
    ): ScenarioRunSummary {
        val vectorLength = scale.madVectorLength
        val inputs = buildMadInputs(scenario.batchMode, vectorLength)
        val readings = inputs.map { buildMadReadings(it) }
        val timingSamples = mutableListOf<InferenceTiming>()
        var lastResult = "n/d"
        var fallbackNote = ""

        suspend fun runCpu() {
            repeat(iterations) {
                coroutineContext.ensureActive()
                val start = System.nanoTime()
                readings.forEach { reading ->
                    val result = getMAD(reading)
                    lastResult = formatMadResult(result)
                }
                val computeMs = nanosToMillis(System.nanoTime() - start)
                timingSamples += InferenceTiming(transferMs = 0.0, computeMs = computeMs)
            }
        }

        if (scenario.delegateMode == DelegateMode.CPU_NATIVE) {
            runCpu()
        } else {
            val delegateType = scenario.delegateMode.delegateType
                ?: error("DelegateType required for ${scenario.displayLabel}")
            var processor: StatModelProcessor? = null
            try {
                processor = StatModelProcessor(
                    context = context,
                    delegateType = delegateType,
                    sizeVector = vectorLength
                )
                repeat(iterations) {
                    coroutineContext.ensureActive()
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
            } catch (ex: Exception) {
                Log.w(TAG, "MAD TFLite falhou para ${vectorLength} (${scenario.displayLabel}): ${ex.message}")
                fallbackNote = " | Fallback CPU (${ex.message ?: "erro"})"
                timingSamples.clear()
                runCpu()
            } finally {
                processor?.close()
            }
        }

        val batchSize = inputs.size
        val timingStats = buildTimingStats(timingSamples)
        val throughput = computeThroughput(
            operations = vectorLength.toDouble() * batchSize,
            durationMs = timingStats.total.mean
        )
        val dataDescription = "${batchSize}×(1 sensor × $vectorLength amostras)"
        val summaryText = buildString {
            appendLine("${scenario.displayLabel} (${scenario.batchMode.displayName}, ${scale.shortLabel})")
            appendLine("Total: ${formatStats(timingStats.total)} | Transfer: ${formatStats(timingStats.transfer)} | Proc: ${formatStats(timingStats.compute)}")
            appendLine("Último resultado: $lastResult$fallbackNote")
        }
        Log.i(TAG, summaryText)

        if (logResults) {
            logScenario(
                context = context,
                scenario = scenario,
                iterations = iterations,
                batchSize = batchSize,
                dataDescription = dataDescription,
                inputSize = vectorLength * batchSize,
                throughput = throughput,
                timingStats = timingStats,
                notes = "$lastResult$fallbackNote | Escala ${scale.shortLabel} | Tempos: ${formatTimingSamples(timingSamples)}",
                scale = scale,
                dataLength = vectorLength
            )
        }

        return ScenarioRunSummary(
            scenario = scenario,
            summaryText = summaryText,
            timings = timingStats,
            notes = lastResult,
            batchSize = batchSize,
            scale = scale
        )
    }

    private suspend fun runFftScenario(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        scale: DataScale,
        logResults: Boolean
    ): ScenarioRunSummary {
        val signalLength = scale.fftSignalLength
        val inputs = buildFftInputs(scenario.batchMode, signalLength)
        val timingSamples = mutableListOf<InferenceTiming>()
        var lastSummary = "-"
        var fallbackNote = ""
        val cpuProcessor = getFftCpuProcessor(signalLength)

        suspend fun runCpu() {
            repeat(iterations) {
                coroutineContext.ensureActive()
                var compute = 0.0
                inputs.forEach { input ->
                    val start = System.nanoTime()
                    val result = cpuProcessor.process(input.samples, input.weights)
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
        }

        if (scenario.delegateMode == DelegateMode.CPU_NATIVE) {
            runCpu()
        } else {
            val delegateType = scenario.delegateMode.delegateType
                ?: error("DelegateType required for ${scenario.displayLabel}")
            var processor: FftTfliteProcessor? = null
            try {
                processor = FftTfliteProcessor(
                    context = context,
                    delegateType = delegateType,
                    numSensors = FFT_NUM_SENSORS,
                    signalLength = signalLength
                )
                repeat(iterations) {
                    coroutineContext.ensureActive()
                    var transfer = 0.0
                    var compute = 0.0
                    inputs.forEach { input ->
                        val result = processor!!.process(input.samples, input.weights)
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
            } catch (ex: Exception) {
                Log.w(TAG, "FFT TFLite falhou para $signalLength (${scenario.displayLabel}): ${ex.message}")
                fallbackNote = " | Fallback CPU (${ex.message ?: "erro"})"
                timingSamples.clear()
                runCpu()
            } finally {
                processor?.close()
            }
        }

        val batchSize = inputs.size
        val timingStats = buildTimingStats(timingSamples)
        val operations = signalLength.toDouble() * FFT_NUM_SENSORS * batchSize
        val throughput = computeThroughput(operations, timingStats.total.mean)
        val dataDescription = "${batchSize}×($FFT_NUM_SENSORS sensores × $signalLength amostras)"
        val summaryText = buildString {
            appendLine("${scenario.displayLabel} (${scenario.batchMode.displayName}, ${scale.shortLabel})")
            appendLine("Total: ${formatStats(timingStats.total)} | Transfer: ${formatStats(timingStats.transfer)} | Proc: ${formatStats(timingStats.compute)}")
            appendLine("Highlights: $lastSummary$fallbackNote")
        }
        Log.i(TAG, summaryText)

        if (logResults) {
            logScenario(
                context = context,
                scenario = scenario,
                iterations = iterations,
                batchSize = batchSize,
                dataDescription = dataDescription,
                inputSize = FFT_NUM_SENSORS * signalLength * batchSize,
                throughput = throughput,
                timingStats = timingStats,
                notes = "$lastSummary$fallbackNote | Escala ${scale.shortLabel} | Tempos: ${formatTimingSamples(timingSamples)}",
                scale = scale,
                dataLength = signalLength
            )
        }

        return ScenarioRunSummary(
            scenario = scenario,
            summaryText = summaryText,
            timings = timingStats,
            notes = "$lastSummary$fallbackNote",
            batchSize = batchSize,
            scale = scale
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
        notes: String,
        scale: DataScale,
        dataLength: Int
    ) {
        ResultLogger.append(
            context,
            scenario.logFileName,
            listOf(
                "Cenário: ${scenario.displayLabel}",
                "Escala: ${scale.shortLabel} ($dataLength pts)",
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

    private fun buildMadInputs(
        batchMode: BatchMode,
        vectorLength: Int
    ): List<Array<IntArray>> {
        return when (batchMode) {
            BatchMode.SINGLE -> listOf(ensureMadInput(vectorLength))
            BatchMode.TEN_PACKETS -> {
                val batch = ensureAccelerometerBatch(vectorLength)
                batch.sensors.take(BATCH_SIZE).map { it.asMadInput() }
            }
        }
    }

    private fun buildFftInputs(
        batchMode: BatchMode,
        signalLength: Int
    ): List<FftInputBuilder.FftProcessorInput> {
        val input = ensureFftInput(signalLength)
        return when (batchMode) {
            BatchMode.SINGLE -> listOf(input)
            BatchMode.TEN_PACKETS -> List(BATCH_SIZE) { input }
        }
    }

    private fun ensureAccelerometerBatch(samplesPerSensor: Int): AccelerometerBatchGenerator.AccelerometerBatch {
        return accelerometerBatches.getOrPut(samplesPerSensor) {
            AccelerometerBatchGenerator.generate(
                numSensors = FFT_NUM_SENSORS,
                samplesPerSensor = samplesPerSensor
            )
        }
    }

    private fun ensureMadInput(vectorLength: Int): Array<IntArray> {
        return madInputsCache.getOrPut(vectorLength) {
            val batch = ensureAccelerometerBatch(vectorLength)
            batch.sensors.first().asMadInput()
        }
    }

    private fun ensureFftInput(signalLength: Int): FftInputBuilder.FftProcessorInput {
        return fftInputsCache.getOrPut(signalLength) {
            val batch = ensureAccelerometerBatch(signalLength)
            FftInputBuilder.fromAccelerometer(batch, signalLength)
        }
    }

    private fun getFftCpuProcessor(signalLength: Int): FftCpuProcessor {
        return fftCpuProcessors.getOrPut(signalLength) {
            FftCpuProcessor(
                numSensors = FFT_NUM_SENSORS,
                signalLength = signalLength
            )
        }
    }

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

    private fun logEnergyResult(
        context: Context,
        scenario: BenchmarkScenario,
        iterations: Int,
        elapsed: Long,
        startSnapshot: BatterySnapshot,
        endSnapshot: BatterySnapshot,
        dropPercent: Double,
        runSummary: ScenarioRunSummary,
        totalStatsText: String,
        scale: DataScale,
        dataLength: String,
        temperatureSummary: String?
    ) {
        val energyLogLines = mutableListOf(
            "Cenário energético: ${scenario.displayLabel}",
            "Escala: ${scale.shortLabel} ($dataLength)",
            "Execuções: $iterations",
            "Tempo: ${formatDuration(elapsed)}",
            "Bateria: ${"%.2f".format(startSnapshot.levelPercent)}% -> ${"%.2f".format(endSnapshot.levelPercent)}% (Δ ${"%.2f".format(dropPercent)}%)"
        )
        temperatureSummary?.let { energyLogLines += it }
        val perfLines = runSummary.summaryText
            .lines()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
        if (perfLines.isNotEmpty()) {
            energyLogLines += "Resumo de desempenho:"
            energyLogLines.addAll(perfLines)
        }
        ResultLogger.append(
            context,
            "energy_tests.txt",
            energyLogLines
        )
        EnergyReporter.append(
            context,
            EnergyLogEntry(
                timestamp = System.currentTimeMillis(),
                scenarioLabel = scenario.displayLabel,
                delegate = scenario.delegateMode.displayName,
                batchMode = scenario.batchMode.displayName,
                durationMinutes = maxOf(1, (elapsed / 60000.0).roundToInt()),
                totalRuns = iterations,
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
                notes = "Tempo ${formatDuration(elapsed)} | Escala ${scale.shortLabel} ($dataLength) | Δ ${"%.2f".format(dropPercent)}% | Total $totalStatsText"
            )
        )
    }

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

    private fun formatDuration(durationMs: Long): String {
        val minutes = TimeUnit.MILLISECONDS.toMinutes(durationMs)
        val seconds =
            TimeUnit.MILLISECONDS.toSeconds(durationMs) - TimeUnit.MINUTES.toSeconds(minutes)
        val millis =
            durationMs - TimeUnit.SECONDS.toMillis(TimeUnit.MILLISECONDS.toSeconds(durationMs))
        return String.format("%02d:%02d.%03d", minutes, seconds, millis)
    }

    private fun nanosToMillis(value: Long): Double = value / 1_000_000.0

    private fun formatDataLength(algorithm: Algorithm, scale: DataScale): String {
        return when (algorithm) {
            Algorithm.MAD -> "${scale.madVectorLength} amostras"
            Algorithm.FFT -> "${scale.fftSignalLength} amostras"
        }
    }

    private fun formatTemperatureRange(start: Double?, end: Double?): String? {
        if (start == null && end == null) return null
        val formattedStart = formatTemperatureValue(start)
        val formattedEnd = formatTemperatureValue(end)
        return "Temperatura: $formattedStart -> $formattedEnd"
    }

    private fun formatTemperatureValue(value: Double?): String =
        value?.let { "%.1f°C".format(it) } ?: "-"

    data class MADResult(
        val mean: Double,
        val stdDev: Double,
        val min: Double,
        val max: Double
    )

    data class BatterySnapshot(
        val levelPercent: Double,
        val isCharging: Boolean,
        val temperatureC: Double?,
        val energyNwh: Double?,
        val chargeMah: Double?,
        val powerSave: Boolean
    )

    data class EnergyScenarioSnapshot(
        val scenario: BenchmarkScenario,
        val summaryLine: String,
        val elapsedMs: Long,
        val dropPercent: Double,
        val totalStats: StatsSummary,
        val scenarioIndex: Int,
        val totalScenarios: Int,
        val percentComplete: Int,
        val scale: DataScale,
        val temperatureStartC: Double?,
        val temperatureEndC: Double?
    )

    companion object {
        private const val TAG = "BenchmarkExecutor"
        const val DEFAULT_BENCH_REPETITIONS = 10
        const val BATCH_SIZE = 10
        const val ENERGY_TEST_ITERATIONS = 100
        const val FFT_NUM_SENSORS = 10
    }
}
