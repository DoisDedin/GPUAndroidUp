package com.example.vulkanfft

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.vulkanfft.util.AccelerometerBatchGenerator
import com.example.vulkanfft.util.BenchmarkProcessor2
import com.example.vulkanfft.util.DelegateType
import com.example.vulkanfft.util.FftCpuProcessor
import com.example.vulkanfft.util.FftInputBuilder
import com.example.vulkanfft.util.FftResult
import com.example.vulkanfft.util.FftTfliteProcessor
import com.example.vulkanfft.util.StatModelProcessor
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

class FirstViewModel : ViewModel() {

    private val _result = MutableLiveData<String>()
    val result: LiveData<String> = _result

    private var benchmarkProcessor: BenchmarkProcessor2? = null

    private var accelerometerBatch: AccelerometerBatchGenerator.AccelerometerBatch? = null
    private var madInput: Array<IntArray>? = null
    private var fftInput: FftInputBuilder.FftProcessorInput? = null
    private val fftCpuProcessor = FftCpuProcessor(
        numSensors = FFT_NUM_SENSORS,
        signalLength = FFT_SIGNAL_LENGTH
    )

    fun runBenchmark(context: Context, delegateType: DelegateType) {
        val input = ensureMadInput()
        viewModelScope.launch {
            benchmarkProcessor = BenchmarkProcessor2(
                processorFactory = {
                    StatModelProcessor(
                        context = context,
                        delegateType = delegateType,
                        sizeVector = MAD_VECTOR_LENGTH
                    )
                }
            )
            val processor = benchmarkProcessor ?: return@launch

            // Chamada do modelo
            val result = processor.runBenchmark(input)
            result.let {
                _result.postValue(
                    "Benchmark (${it.mean}ms média | ${it.stdDev}ms desv.)\nMin: ${it.min}ms | Max: ${it.max}ms"
                )
            } ?: run {
                _result.postValue("Benchmark não inicializado.")
            }
        }
    }

    fun runCPU() {
        val input = ensureMadInput()
        viewModelScope.launch {
            val timestamps = input[0]
            val x = input[1]
            val y = input[2]
            val z = input[3]
            val readings = timestamps.indices.map { i ->
                AccelerometerSample(
                    timestamp = timestamps[i].toLong(),
                    x = x[i],
                    y = y[i],
                    z = z[i]
                )
            }

            val result = getMAD(readings)

            Log.d("MAD" , "SEND")
            result.let {
                _result.postValue(
                    "Benchmark Mad: (${it.mean}  média | ${it.stdDev}ms desv.)\nMin: ${it.min}ms | Max: ${it.max}ms"
                )
            }
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

    fun closeProcessor() {
        benchmarkProcessor = null
    }

    fun runFftCpu() {
        viewModelScope.launch {
            val input = ensureFftInput()

            val start = System.nanoTime()
            val result = fftCpuProcessor.process(input.samples, input.weights)
            val durationMs = (System.nanoTime() - start) / 1_000_000.0

            _result.postValue(formatFftSummary("FFT CPU", durationMs, result))
        }
    }

    fun runFftTflite(context: Context, delegateType: DelegateType) {
        viewModelScope.launch {
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

            _result.postValue(
                formatFftSummary("FFT TFLite (${delegateType.name})", durationMs, result)
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
}
