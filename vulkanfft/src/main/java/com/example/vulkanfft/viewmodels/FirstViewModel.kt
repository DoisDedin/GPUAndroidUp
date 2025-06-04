package com.example.vulkanfft.viewmodels

import android.content.Context
import android.util.Log
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.vulkanfft.util.BenchmarkProcessor
import com.example.vulkanfft.util.BenchmarkProcessor2
import com.example.vulkanfft.util.StatModelProcessor
import com.example.vulkanfft.util.SumProcessorGPU
import kotlinx.coroutines.launch
import java.util.concurrent.TimeUnit
import kotlin.math.sqrt

class FirstViewModel : ViewModel() {

    private val _result = MutableLiveData<String>()
    val result: LiveData<String> = _result

    private var benchmarkProcessor: BenchmarkProcessor2? = null

    private var inputData: Array<IntArray> = arrayOf()

    private fun generateInputData() {
        val N = 4096
        val timestamps = IntArray(N) { it * 20 }
        val x = IntArray(N) { (0..9).random() }
        val y = IntArray(N) { (0..9).random() }
        val z = IntArray(N) { (0..9).random() }
        inputData = arrayOf(timestamps, x, y, z)
    }

    fun runBenchmark(context: Context, delegateType: SumProcessorGPU.DelegateType) {
        if (inputData.isEmpty()) generateInputData()
        viewModelScope.launch {
            benchmarkProcessor = BenchmarkProcessor2(
                processorFactory = {
                    StatModelProcessor(
                        context = context,
                        delegateType = delegateType,
                        sizeVector = 4096
                    )
                }
            )
            val processor = benchmarkProcessor ?: return@launch

            // Chamada do modelo
            val result = processor.runBenchmark(inputData)
            result.let {
                _result.postValue(
                    "Benchmark (${it.mean}ms média | ${it.stdDev}ms desv.)\nMin: ${it.min}ms | Max: ${it.max}ms"
                )
            } ?: run {
                _result.postValue("Benchmark não inicializado.")
            }
        }
    }

    fun runCPU(context: Context) {
        if (inputData.isEmpty()) generateInputData()
        viewModelScope.launch {

            val readings = inputData.indices.map { i ->
                AccelerometerSample(
                    timestamp = inputData[0][i].toLong(),
                    x = inputData[1][i],
                    y = inputData[2][i],
                    z = inputData[3][i]
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


    private fun getMAD(readings: List<AccelerometerSample>): MADResult{

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
}