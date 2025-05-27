package com.example.vulkanfft.viewmodels

import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.vulkanfft.util.BenchmarkProcessor
import com.example.vulkanfft.util.SumProcessorGPU
import kotlinx.coroutines.launch

class FirstViewModel : ViewModel() {

    private val _result = MutableLiveData<String>()
    val result: LiveData<String> = _result

    private var benchmarkProcessor: BenchmarkProcessor? = null

    fun runBenchmark(context: Context, delegateType: SumProcessorGPU.DelegateType) {
        viewModelScope.launch {
            benchmarkProcessor = BenchmarkProcessor(
                processorFactory = {
                    SumProcessorGPU(
                        context = context,
                        delegateType = delegateType
                    )
                }
            )
            val result = benchmarkProcessor?.runBenchmark(x1 = 20.0F, x2 = 2.0F)

            result?.let {
                _result.postValue(
                    "Benchmark (${it.mean}ms média | ${it.stdDev}ms desv.)\nMin: ${it.min}ms | Max: ${it.max}ms"
                )
            } ?: run {
                _result.postValue("Benchmark não inicializado.")
            }
        }

    }

    fun closeProcessor() {
        benchmarkProcessor = null
    }
}