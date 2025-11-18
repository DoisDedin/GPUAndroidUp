package com.example.vulkanfft.util

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
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
    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(sizeVector * INPUT_AXES * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())
    private val inputFloatBuffer = inputBuffer.asFloatBuffer()
    private val outputTensor = FloatArray(4)

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
        interpreter = try {
            Interpreter(modelBuffer, options)
        } catch (e: IllegalArgumentException) {
            Log.w(tag, "Delegate ${delegateType.name} indisponível (${e.message}). Usando CPU/XNNPACK.")
            gpuDelegate?.close()
            gpuDelegate = null
            nnapiDelegate?.close()
            nnapiDelegate = null
            val fallbackOptions = Interpreter.Options().apply { setUseXNNPACK(true) }
            Interpreter(modelBuffer, fallbackOptions)
        }

        interpreter.resizeInput(0, intArrayOf(sizeVector, 3))
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

    fun process(input: Array<IntArray>): InferenceResult<FloatArray> {
        Log.d(tag, "===== Início do processamento de estatísticas =====")

        val transferStart = System.nanoTime()
        inputFloatBuffer.rewind()
        val x = input[1]
        val y = input[2]
        val z = input[3]
        for (i in 0 until sizeVector) {
            inputFloatBuffer.put(x[i].toFloat())
            inputFloatBuffer.put(y[i].toFloat())
            inputFloatBuffer.put(z[i].toFloat())
        }
        inputBuffer.rewind()
        val transferDuration = nanosToMillis(System.nanoTime() - transferStart)

        val computeStart = System.nanoTime()
        interpreter.run(inputBuffer, outputTensor)
        val computeDuration = nanosToMillis(System.nanoTime() - computeStart)

        Log.d(tag, "Transferência: ${"%.3f".format(transferDuration)} ms")
        Log.d(tag, "Duração da inferência: ${"%.3f".format(computeDuration)} ms")
        Log.d(tag, "MAD calculado: ${outputTensor[0]}")
        Log.d(tag, "===== Fim do processamento =====")

        return InferenceResult(
            output = outputTensor.copyOf(),
            timing = InferenceTiming(
                transferMs = transferDuration,
                computeMs = computeDuration
            )
        )
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
        nnapiDelegate?.close()
    }

    private fun nanosToMillis(value: Long): Double = value / 1_000_000.0

    companion object {
        private const val FLOAT_BYTES = 4
        private const val INPUT_AXES = 3
    }
}
