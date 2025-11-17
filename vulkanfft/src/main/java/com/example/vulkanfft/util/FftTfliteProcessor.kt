package com.example.vulkanfft.util

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class FftTfliteProcessor(
    context: Context,
    delegateType: DelegateType,
    private val numSensors: Int = 10,
    private val signalLength: Int = 10
) {

    private val tag = "FftTfliteProcessor"
    private val freqBins = signalLength / 2 + 1
    private var interpreter: Interpreter
    private var gpuDelegate: Delegate? = null
    private var nnapiDelegate: Delegate? = null

    init {
        val options = Interpreter.Options()

        when (delegateType) {
            DelegateType.GPU -> {
                val gpuOptions = GpuDelegate.Options().apply {
                    setPrecisionLossAllowed(false)
                    setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                }
                gpuDelegate = GpuDelegate(gpuOptions).also { options.addDelegate(it) }
                Log.d(tag, "Usando delegate GPU (full precision)")
            }

            DelegateType.NNAPI -> {
                nnapiDelegate = NnApiDelegate().also { options.addDelegate(it) }
                Log.d(tag, "Usando delegate NNAPI")
            }

            DelegateType.CPU -> {
                options.setUseXNNPACK(true)
                Log.d(tag, "Usando delegate CPU (XNNPACK)")
            }
        }

        val modelBuffer = loadModelFile(context, "fft_model.tflite")
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

    fun process(
        samples: Array<FloatArray>,
        weights: Array<FloatArray>
    ): FftResult {
        require(samples.size == numSensors) { "Esperado $numSensors sensores" }
        require(weights.size == numSensors) { "Esperado $numSensors vetores de peso" }

        samples.forEachIndexed { index, floats ->
            require(floats.size == signalLength) {
                "Sensor #$index deveria ter $signalLength amostras"
            }
        }

        weights.forEachIndexed { index, floats ->
            require(floats.size == freqBins) {
                "Vetor de peso do sensor #$index deveria ter $freqBins posições"
            }
        }

        // O conversor exporta as entradas na ordem [weights, samples],
        // então respeitamos essa convenção para evitar reshapes incorretos.
        val inputs = arrayOf(weights, samples)
        val floatCount = numSensors * freqBins * 4
        val outputBuffer = ByteBuffer.allocateDirect(4 * floatCount).order(ByteOrder.nativeOrder())
        val outputs = hashMapOf<Int, Any>(0 to outputBuffer)

        val start = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputs, outputs)
        val durationMs = (System.nanoTime() - start) / 1_000_000.0
        Log.d(tag, "Inferência FFT TFLite executada em ${"%.3f".format(durationMs)} ms")

        outputBuffer.rewind()
        val floatBuffer = outputBuffer.asFloatBuffer()

        val complexSpectrum = Array(numSensors) {
            Array(freqBins) { ComplexFloat(0f, 0f) }
        }
        val magnitudes = Array(numSensors) { FloatArray(freqBins) }
        val weightedMagnitudes = Array(numSensors) { FloatArray(freqBins) }

        for (sensor in 0 until numSensors) {
            for (bin in 0 until freqBins) {
                val real = floatBuffer.get()
                val imag = floatBuffer.get()
                val magnitude = floatBuffer.get()
                val weighted = floatBuffer.get()

                complexSpectrum[sensor][bin] = ComplexFloat(real, imag)
                magnitudes[sensor][bin] = magnitude
                weightedMagnitudes[sensor][bin] = weighted
            }
        }

        return FftResult(
            complexSpectrum = complexSpectrum,
            magnitudes = magnitudes,
            weightedMagnitudes = weightedMagnitudes
        )
    }

    fun close() {
        interpreter.close()
        (gpuDelegate as? GpuDelegate)?.close()
        (nnapiDelegate as? NnApiDelegate)?.close()
    }
}
