package com.example.vulkanfft

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.vulkanfft.logging.BenchmarkEntry
import com.example.vulkanfft.logging.BenchmarkReporter
import com.example.vulkanfft.logging.DeviceInfoProvider
import com.example.vulkanfft.logging.ResultLogger
import com.example.vulkanfft.util.AccelerometerBatchGenerator
import com.example.vulkanfft.util.BenchmarkProcessor2
import com.example.vulkanfft.util.DelegateType
import com.example.vulkanfft.util.FftCpuProcessor
import com.example.vulkanfft.util.FftInputBuilder
import com.example.vulkanfft.util.FftResult
import com.example.vulkanfft.util.FftTfliteProcessor
import com.example.vulkanfft.util.StatModelProcessor
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

class FirstViewModel : ViewModel() {

    private val _result = MutableLiveData<String>()
    val result: LiveData<String> = _result

    private val _progress = MutableLiveData<BenchmarkProgress>()
    val progress: LiveData<BenchmarkProgress> = _progress

    private var accelerometerBatch: AccelerometerBatchGenerator.AccelerometerBatch? = null
    private var madInput: Array<IntArray>? = null
    private var fftInput: FftInputBuilder.FftProcessorInput? = null
    private val fftCpuProcessor = FftCpuProcessor(
        numSensors = FFT_NUM_SENSORS,
        signalLength = FFT_SIGNAL_LENGTH
    )

    fun runMadBenchmark(
        context: Context,
        delegateType: DelegateType,
        repetitions: Int = DEFAULT_BENCH_REPETITIONS
    ) {
        viewModelScope.launch(Dispatchers.Default) {
            val summary = executeMadBenchmark(context, delegateType, repetitions)
            _result.postValue(summary)
        }
    }

    fun runMadCpuSingle(context: Context) {
        viewModelScope.launch(Dispatchers.Default) {
            val summary = executeMadCpuSingle(context)
            _result.postValue(summary)
        }
    }

    fun runMadCpuBatch(context: Context, iterations: Int = 10) {
        viewModelScope.launch(Dispatchers.Default) {
            val summary = executeMadCpuBatch(context, iterations)
            _result.postValue(summary)
        }
    }

    fun runMadDelegateSingle(context: Context, delegateType: DelegateType) {
        viewModelScope.launch(Dispatchers.Default) {
            val summary = executeMadDelegateSingle(context, delegateType)
            _result.postValue(summary)
        }
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
        // Agrupar por janelas de 5 segundos (5000 ms)
        val windowSizeMillis = TimeUnit.SECONDS.toNanos(5)
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
        Log.d("MAD", "SizeBlocks: ${blockAverages.size}")
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

    fun runFftCpu(context: Context) {
        viewModelScope.launch(Dispatchers.Default) {
            val summary = executeFftCpu(context)
            _result.postValue(summary)
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

    private suspend fun executeMadCpuSingle(context: Context): String {
        val input = ensureMadInput()
        val readings = buildMadReadings(input)
        val start = System.nanoTime()
        val result = getMAD(readings)
        val durationMs = (System.nanoTime() - start) / 1_000_000.0
        val summary =
            "MAD CPU -> (${result.mean} média | ${result.stdDev} desv.)\nMin: ${result.min} | Max: ${result.max}"
        ResultLogger.append(
            context,
            "mad_cpu.txt",
            listOf(summary, "Tamanho janela: $MAD_VECTOR_LENGTH")
        )
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = "MAD CPU Kotlin",
                processingMode = "Kotlin",
                delegate = "CPU",
                dataDescription = "1 sensor × $MAD_VECTOR_LENGTH amostras",
                inputSize = MAD_VECTOR_LENGTH,
                durationMs = durationMs,
                throughput = computeThroughput(MAD_VECTOR_LENGTH.toDouble(), durationMs),
                estimatedEnergyImpact = "Alta (CPU dedicada)",
                deviceInfo = DeviceInfoProvider.collect(context),
                extraNotes = summary
            )
        )
        return summary
    }

    private suspend fun executeMadCpuBatch(context: Context, iterations: Int): String {
        val input = ensureMadInput()
        val readings = buildMadReadings(input)
        val durations = mutableListOf<Double>()
        repeat(iterations) {
            val start = System.nanoTime()
            getMAD(readings)
            durations += (System.nanoTime() - start) / 1_000_000.0
        }
        val mean = durations.average()
        val summary =
            "MAD CPU x$iterations -> média ${"%.3f".format(mean)} ms | min ${"%.3f".format(durations.minOrNull() ?: 0.0)} | max ${"%.3f".format(durations.maxOrNull() ?: 0.0)}"
        ResultLogger.append(
            context,
            "mad_cpu_batch.txt",
            listOf(summary, "Execuções: ${durations.joinToString()}")
        )
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = "MAD CPU Kotlin x$iterations",
                processingMode = "Kotlin",
                delegate = "CPU",
                dataDescription = "1 sensor × $MAD_VECTOR_LENGTH amostras",
                inputSize = MAD_VECTOR_LENGTH,
                durationMs = mean,
                throughput = computeThroughput(MAD_VECTOR_LENGTH.toDouble(), mean),
                estimatedEnergyImpact = "Alta (CPU dedicada)",
                deviceInfo = DeviceInfoProvider.collect(context),
                extraNotes = "Execuções: ${durations.joinToString()}"
            )
        )
        return summary
    }

    private suspend fun executeMadDelegateSingle(
        context: Context,
        delegateType: DelegateType
    ): String {
        val input = ensureMadInput()
        val processor = StatModelProcessor(
            context = context,
            delegateType = delegateType,
            sizeVector = MAD_VECTOR_LENGTH
        )
        val start = System.nanoTime()
        val output = processor.process(input)
        val durationMs = (System.nanoTime() - start) / 1_000_000.0
        processor.close()
        val stats =
            "Mean=${output.getOrNull(0)}, Std=${output.getOrNull(1)}, Min=${output.getOrNull(2)}, Max=${output.getOrNull(3)}"
        val summary =
            "MAD TFLite (${delegateType.name}) -> ${"%.3f".format(durationMs)} ms\n$stats"
        ResultLogger.append(
            context,
            "mad_single.txt",
            listOf("Delegate: ${delegateType.name}", summary)
        )
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = "MAD TFLite (${delegateType.name})",
                processingMode = "TFLite",
                delegate = delegateType.name,
                dataDescription = "1 sensor × $MAD_VECTOR_LENGTH amostras",
                inputSize = MAD_VECTOR_LENGTH,
                durationMs = durationMs,
                throughput = computeThroughput(MAD_VECTOR_LENGTH.toDouble(), durationMs),
                estimatedEnergyImpact = estimateEnergy(delegateType),
                deviceInfo = DeviceInfoProvider.collect(context),
                extraNotes = stats
            )
        )
        return summary
    }

    private suspend fun executeMadBenchmark(
        context: Context,
        delegateType: DelegateType,
        repetitions: Int
    ): String {
        val input = ensureMadInput()
        val processor = BenchmarkProcessor2(
            processorFactory = {
                StatModelProcessor(
                    context = context,
                    delegateType = delegateType,
                    sizeVector = MAD_VECTOR_LENGTH
                )
            },
            repetitions = repetitions
        )
        val result = processor.runBenchmark(input)
        val summary =
            "Benchmark ${delegateType.name} (${result.mean}ms média | ${result.stdDev}ms desv.)\nMin: ${result.min}ms | Max: ${result.max}ms"
        ResultLogger.append(
            context,
            "mad_benchmark.txt",
            listOf(
                "Delegate: ${delegateType.name}",
                "Média: ${result.mean}",
                "Desvio padrão: ${result.stdDev}",
                "Min/Max: ${result.min} / ${result.max}",
                "Amostras: ${result.durations.joinToString()}"
            )
        )
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = "MAD Benchmark (${delegateType.name})",
                processingMode = "TFLite Benchmark",
                delegate = delegateType.name,
                dataDescription = "1 sensor × $MAD_VECTOR_LENGTH amostras",
                inputSize = MAD_VECTOR_LENGTH,
                durationMs = result.mean,
                throughput = computeThroughput(
                    MAD_VECTOR_LENGTH.toDouble(),
                    result.mean
                ),
                estimatedEnergyImpact = estimateEnergy(delegateType),
                deviceInfo = DeviceInfoProvider.collect(context),
                extraNotes = "Execuções: ${result.durations.joinToString()}"
            )
        )
        return summary
    }

    private suspend fun executeFftCpu(context: Context): String {
        val input = ensureFftInput()
        val start = System.nanoTime()
        val result = fftCpuProcessor.process(input.samples, input.weights)
        val durationMs = (System.nanoTime() - start) / 1_000_000.0
        val summary = formatFftSummary("FFT CPU", durationMs, result)
        ResultLogger.append(
            context,
            "fft_cpu.txt",
            listOf(summary)
        )
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = "FFT CPU",
                processingMode = "Kotlin DFT",
                delegate = "CPU",
                dataDescription = "$FFT_NUM_SENSORS sensores × $FFT_SIGNAL_LENGTH amostras",
                inputSize = FFT_NUM_SENSORS * FFT_SIGNAL_LENGTH,
                durationMs = durationMs,
                throughput = computeThroughput(
                    (FFT_NUM_SENSORS * FFT_SIGNAL_LENGTH).toDouble(),
                    durationMs
                ),
                estimatedEnergyImpact = "Alta (CPU)",
                deviceInfo = DeviceInfoProvider.collect(context),
                extraNotes = summary
            )
        )
        return summary
    }

    private suspend fun executeFftTflite(
        context: Context,
        delegateType: DelegateType
    ): String {
        val input = ensureFftInput()
        val processor = FftTfliteProcessor(
            context = context,
            delegateType = delegateType,
            numSensors = FFT_NUM_SENSORS,
            signalLength = FFT_SIGNAL_LENGTH
        )
        val start = System.nanoTime()
        val result = processor.process(input.samples, input.weights)
        val durationMs = (System.nanoTime() - start) / 1_000_000.0
        processor.close()
        val summary =
            formatFftSummary("FFT TFLite (${delegateType.name})", durationMs, result)
        ResultLogger.append(
            context,
            "fft_tflite.txt",
            listOf(summary, "Delegate: ${delegateType.name}")
        )
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = "FFT TFLite (${delegateType.name})",
                processingMode = "TFLite",
                delegate = delegateType.name,
                dataDescription = "$FFT_NUM_SENSORS sensores × $FFT_SIGNAL_LENGTH amostras",
                inputSize = FFT_NUM_SENSORS * FFT_SIGNAL_LENGTH,
                durationMs = durationMs,
                throughput = computeThroughput(
                    (FFT_NUM_SENSORS * FFT_SIGNAL_LENGTH).toDouble(),
                    durationMs
                ),
                estimatedEnergyImpact = estimateEnergy(delegateType),
                deviceInfo = DeviceInfoProvider.collect(context),
                extraNotes = summary
            )
        )
        return summary
    }

    private suspend fun executeFftBenchmark(
        context: Context,
        delegateType: DelegateType,
        repetitions: Int = DEFAULT_BENCH_REPETITIONS,
        warmup: Int = 1
    ): String {
        val input = ensureFftInput()
        val processor = FftTfliteProcessor(
            context = context,
            delegateType = delegateType,
            numSensors = FFT_NUM_SENSORS,
            signalLength = FFT_SIGNAL_LENGTH
        )
        val durations = mutableListOf<Double>()
        val totalRuns = repetitions + warmup
        repeat(totalRuns) { run ->
            val start = System.nanoTime()
            processor.process(input.samples, input.weights)
            val durationMs = (System.nanoTime() - start) / 1_000_000.0
            if (run >= warmup) {
                durations += durationMs
            }
        }
        processor.close()
        val mean = durations.average()
        val min = durations.minOrNull() ?: mean
        val max = durations.maxOrNull() ?: mean
        val summary =
            "FFT Benchmark (${delegateType.name}) -> média ${"%.3f".format(mean)} ms | min ${"%.3f".format(min)} | max ${"%.3f".format(max)}"
        ResultLogger.append(
            context,
            "fft_benchmark.txt",
            listOf(
                summary,
                "Execuções (${repetitions}): ${durations.joinToString()}"
            )
        )
        BenchmarkReporter.append(
            context,
            BenchmarkEntry(
                timestamp = System.currentTimeMillis(),
                testName = "FFT Benchmark (${delegateType.name})",
                processingMode = "TFLite Benchmark",
                delegate = delegateType.name,
                dataDescription = "$FFT_NUM_SENSORS sensores × $FFT_SIGNAL_LENGTH amostras",
                inputSize = FFT_NUM_SENSORS * FFT_SIGNAL_LENGTH,
                durationMs = mean,
                throughput = computeThroughput(
                    (FFT_NUM_SENSORS * FFT_SIGNAL_LENGTH).toDouble(),
                    mean
                ),
                estimatedEnergyImpact = estimateEnergy(delegateType),
                deviceInfo = DeviceInfoProvider.collect(context),
                extraNotes = "Execuções (${repetitions}): ${durations.joinToString()}"
            )
        )
        return summary
    }

    private fun computeThroughput(operations: Double, durationMs: Double): Double {
        if (durationMs <= 0.0) return 0.0
        return operations / (durationMs / 1000.0)
    }

    private fun estimateEnergy(delegateType: DelegateType): String {
        return when (delegateType) {
            DelegateType.CPU -> "Média (CPU otimizada)"
            DelegateType.GPU -> "Baixa (GPU paralela)"
            DelegateType.NNAPI -> "Baixa/Média (NNAPI)"
        }
    }

    fun runFftTflite(context: Context, delegateType: DelegateType) {
        viewModelScope.launch(Dispatchers.Default) {
            val summary = executeFftTflite(context, delegateType)
            _result.postValue(summary)
        }
    }

    fun runFullBenchmarkSuite(context: Context) {
        viewModelScope.launch(Dispatchers.Default) {
        val tasks = listOf(
            TestTask("MAD CPU Kotlin") { executeMadCpuSingle(context) },
            TestTask("MAD CPU Kotlin x10") { executeMadCpuBatch(context, 10) },
            TestTask("MAD TFLite CPU") { executeMadDelegateSingle(context, DelegateType.CPU) },
            TestTask("MAD TFLite GPU") { executeMadDelegateSingle(context, DelegateType.GPU) },
            TestTask("MAD TFLite NNAPI") { executeMadDelegateSingle(context, DelegateType.NNAPI) },
            TestTask("MAD Benchmark CPU") {
                executeMadBenchmark(context, DelegateType.CPU, DEFAULT_BENCH_REPETITIONS)
            },
            TestTask("MAD Benchmark GPU") {
                executeMadBenchmark(context, DelegateType.GPU, DEFAULT_BENCH_REPETITIONS)
            },
            TestTask("MAD Benchmark NNAPI") {
                executeMadBenchmark(context, DelegateType.NNAPI, DEFAULT_BENCH_REPETITIONS)
            },
            TestTask("FFT CPU") { executeFftCpu(context) },
            TestTask("FFT TFLite CPU") { executeFftTflite(context, DelegateType.CPU) },
            TestTask("FFT TFLite GPU") { executeFftTflite(context, DelegateType.GPU) },
            TestTask("FFT TFLite NNAPI") { executeFftTflite(context, DelegateType.NNAPI) },
            TestTask("FFT Benchmark CPU") {
                executeFftBenchmark(context, DelegateType.CPU, DEFAULT_BENCH_REPETITIONS)
            },
            TestTask("FFT Benchmark GPU") {
                executeFftBenchmark(context, DelegateType.GPU, DEFAULT_BENCH_REPETITIONS)
            },
            TestTask("FFT Benchmark NNAPI") {
                executeFftBenchmark(context, DelegateType.NNAPI, DEFAULT_BENCH_REPETITIONS)
            }
        )
            _progress.postValue(
                BenchmarkProgress(
                    total = tasks.size,
                    current = 0,
                    running = true,
                    message = "Iniciando suíte de testes"
                )
            )
            tasks.forEachIndexed { index, task ->
                val summary = task.block()
                _result.postValue(summary)
                _progress.postValue(
                    BenchmarkProgress(
                        total = tasks.size,
                        current = index + 1,
                        running = index + 1 < tasks.size,
                        message = "Executado: ${task.label}"
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

    private fun formatFftSummary(
        label: String,
        durationMs: Double,
        result: FftResult
    ): String {
        val totals = result.weightedMagnitudes.mapIndexed { index, weights ->
            "S$index=${"%.2f".format(weights.sum())}"
        }.take(4) // mantemos a string curta
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

    companion object {
        private const val FFT_NUM_SENSORS = 10
        private const val FFT_SIGNAL_LENGTH = 4096
        private const val MAD_VECTOR_LENGTH = 4096
        private const val DEFAULT_BENCH_REPETITIONS = 10
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

    private data class TestTask(
        val label: String,
        val block: suspend () -> String
    )
}
