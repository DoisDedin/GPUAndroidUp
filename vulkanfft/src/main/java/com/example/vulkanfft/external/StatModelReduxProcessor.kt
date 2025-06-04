package com.example.vulkanfft.external

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Classe responsável por carregar o modelo TFLite mad_model.tflite
 * e executar o cálculo estatístico com delegate CPU.
 *
 * A entrada deve ter o formato [4][sizeVector]:
 * - input[0] = timestamps (int32)
 * - input[1] = eixo X do acelerômetro (int32)
 * - input[2] = eixo Y do acelerômetro (int32)
 * - input[3] = eixo Z do acelerômetro (int32)
 *
 * A saída será um vetor float[4] com: [mean, stdDev, min, max]
 */
class StatModelReduxProcessor(
    context: Context,
    private val sizeVector: Int
) {
    private val tag = "StatModelProcessor"
    private var interpreter: Interpreter

    init {
        val options = Interpreter.Options().apply {
            useXNNPACK = true
        }

        val modelBuffer = loadModelFile(context)
        interpreter = Interpreter(modelBuffer, options)

        interpreter.resizeInput(0, intArrayOf(4, sizeVector))
        interpreter.allocateTensors()
        Log.d(tag, "✅ Modelo carregado com input shape [4, $sizeVector] e delegate CPU (XNNPACK)")
    }

    private fun loadModelFile(context: Context, filename: String = "mad_model.tflite"): ByteBuffer {
        val fileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Executa o modelo com os dados de entrada [4][N] e retorna as estatísticas
     */
    fun process(input: Array<IntArray>): FloatArray {
        require(input.size == 4 && input.all { it.size == sizeVector }) {
            "Formato de entrada inválido: esperado [4][$sizeVector]"
        }

        val outputTensor = FloatArray(4)

        val startTime = System.nanoTime()
        interpreter.run(input, outputTensor)
        val endTime = System.nanoTime()

        val durationMs = (endTime - startTime) / 1_000_000.0
        Log.d(tag, "Duração da inferência: $durationMs ms")
        Log.d(tag, "Saída: ${outputTensor.joinToString(", ")}")

        return outputTensor
    }

    fun close() {
        interpreter.close()
    }
}