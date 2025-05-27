package com.example.vulkanfft.util

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class FFTProcessorGPU(context: Context) {

    private val tag = "DistanceProcessorGPU"
    private var interpreter: Interpreter
    private var gpuDelegate: GpuDelegate

    init {
        gpuDelegate = GpuDelegate()
        val options = Interpreter.Options().addDelegate(gpuDelegate)

        fun loadModelFile(context: Context, filename: String): ByteBuffer {
            val assetFileDescriptor = context.assets.openFd(filename)
            val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
            val fileChannel = fileInputStream.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }

        val modelBuffer = loadModelFile(context, "sum_model_16bit.tflite")
        interpreter = Interpreter(modelBuffer, options)
    }

    fun calculateDistanceGPU(input: FloatArray): FloatArray {
        Log.d(tag, "====== Início do cálculo GPU ======")
        val startTime = System.currentTimeMillis()

        val inputSize = input.size
        if (inputSize < 2) {
            Log.e(tag, "Input insuficiente para cálculo de distância.")
            return FloatArray(0)
        }

        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize)
            .order(ByteOrder.nativeOrder())
        input.forEach { inputBuffer.putFloat(it) }
        inputBuffer.rewind()

        val outputBuffer = ByteBuffer.allocateDirect(4 * (inputSize - 1))
            .order(ByteOrder.nativeOrder())

        interpreter.run(inputBuffer, outputBuffer)

        outputBuffer.rewind()
        val output = FloatArray(inputSize - 1)
        outputBuffer.asFloatBuffer().get(output)

        val endTime = System.currentTimeMillis()
        val duration = endTime - startTime

        Log.d(tag, "Resultado GPU: ${output.joinToString()}")
        Log.d(tag, "Tempo total GPU (ms): $duration")
        Log.d(tag, "====== Fim do cálculo GPU ======")

        return output
    }

    fun close() {
        interpreter.close()
        gpuDelegate.close()
    }
}