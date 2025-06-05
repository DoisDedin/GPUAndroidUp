package com.example.vulkanfft.util

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

class StatModelProcessor(
    context: Context,
    val delegateType: DelegateType,
    private val sizeVector: Int
) {
    private val tag = "StatModelProcessor"
    private var interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: org.tensorflow.lite.nnapi.NnApiDelegate? = null

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
                    nnapiDelegate = org.tensorflow.lite.nnapi.NnApiDelegate()
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
        val modelBuffer = loadModelFile(context, "mad_model.tflite")
        interpreter = Interpreter(modelBuffer, options)

        // ⬇️ Configure o shape aqu
        interpreter.resizeInput(0, intArrayOf(4, sizeVector))
        interpreter.allocateTensors()
    }

    private fun loadModelFile(context: Context, filename: String): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun process(input: Array<IntArray>): FloatArray {
        Log.d(tag, "===== Início do processamento de estatísticas =====")
        // input já é [4][N], então passamos ele diretamente
        val outputTensor = FloatArray(4)

        val startTime = System.nanoTime()
        interpreter.run(input, outputTensor)
        val endTime = System.nanoTime()

        val durationMs = (endTime - startTime) / 1_000_000.0
        Log.d(tag, "Duração da inferência: $durationMs ms")
        Log.d(tag, "MAD calculado: ${outputTensor[0]}")
        Log.d(tag, "===== Fim do processamento =====")

        return outputTensor
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        nnapiDelegate?.close()
    }
}
