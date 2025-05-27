package com.example.vulkanfft.util

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class SumProcessorGPU(
    context: Context,
    val delegateType: DelegateType
) {

    private val tag = "SumProcessorManual"
    private var interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null

    enum class DelegateType {
        GPU, NNAPI, CPU
    }

    init {
        val options = Interpreter.Options()

        when (delegateType) {
            DelegateType.GPU -> {
                try {
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                    Log.d(tag, "✅ Usando GPU Delegate")
                } catch (e: Exception) {
                    throw RuntimeException("GPU Delegate não suportado neste dispositivo: ${e.message}")
                }
            }

            DelegateType.NNAPI -> {
                try {
                    nnapiDelegate = NnApiDelegate()
                    options.addDelegate(nnapiDelegate)
                    Log.d(tag, "✅ Usando NNAPI Delegate")
                } catch (e: Exception) {
                    throw RuntimeException("NNAPI Delegate não suportado neste dispositivo: ${e.message}")
                }
            }

            DelegateType.CPU -> {
                options.setUseXNNPACK(true)
                Log.d(tag, "✅ Usando CPU com XNNPACK")
            }
        }

        val modelBuffer = loadModelFile(context, "sum_model_16bit.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }

    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    suspend fun calculateSum(x1: Float, x2: Float): Float {
        Log.d(tag, "===== Início do cálculo de soma =====")
        Log.d(tag, "Rodando com Delegate: $delegateType")

        val input = Array(1) { floatArrayOf(x1, x2) }
        val output = Array(1) { FloatArray(1) }

        val startTime = System.nanoTime()
        interpreter.run(input, output)
        val endTime = System.nanoTime()

        val durationMs = (endTime - startTime) / 1_000_000.0
        val result = output[0][0]

        Log.d(tag, "Resultado da soma: $result")
        Log.d(tag, "Duração: $durationMs ms")
        Log.d(tag, "===== Fim do cálculo =====")
        return result
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        nnapiDelegate?.close()
    }
}
