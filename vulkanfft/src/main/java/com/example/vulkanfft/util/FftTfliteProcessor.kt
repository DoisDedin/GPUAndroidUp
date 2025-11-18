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
    private val weightsInputBuffer = ByteBuffer.allocateDirect(numSensors * freqBins * FLOAT_BYTES)
        .order(ByteOrder.nativeOrder())
    private val weightsFloatBuffer = weightsInputBuffer.asFloatBuffer()
    private val samplesInputBuffer = ByteBuffer.allocateDirect(numSensors * signalLength * FLOAT_BYTES)
        .order(ByteOrder.nativeOrder())
    private val samplesFloatBuffer = samplesInputBuffer.asFloatBuffer()
    private val outputBuffer = ByteBuffer.allocateDirect(numSensors * freqBins * OUTPUT_FIELDS * FLOAT_BYTES)
        .order(ByteOrder.nativeOrder())
    private val outputFloatBuffer = outputBuffer.asFloatBuffer()
    private val outputs = hashMapOf<Int, Any>(0 to outputBuffer)

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
    ): InferenceResult<FftResult> {
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

        val transferStart = System.nanoTime()
        weightsFloatBuffer.rewind()
        for (sensor in 0 until numSensors) {
            val sensorWeights = weights[sensor]
            for (bin in 0 until freqBins) {
                weightsFloatBuffer.put(sensorWeights[bin])
            }
        }
        samplesFloatBuffer.rewind()
        for (sensor in 0 until numSensors) {
            val sensorSamples = samples[sensor]
            for (sample in 0 until signalLength) {
                samplesFloatBuffer.put(sensorSamples[sample])
            }
        }
        weightsInputBuffer.rewind()
        samplesInputBuffer.rewind()
        val transferDuration = nanosToMillis(System.nanoTime() - transferStart)

        val computeStart = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(
            arrayOf(weightsInputBuffer, samplesInputBuffer),
            outputs
        )
        val computeDuration = nanosToMillis(System.nanoTime() - computeStart)
        Log.d(tag, "Transferência FFT: ${"%.3f".format(transferDuration)} ms")
        Log.d(tag, "Processamento FFT: ${"%.3f".format(computeDuration)} ms")

        outputBuffer.rewind()
        outputFloatBuffer.rewind()

        val complexSpectrum = Array(numSensors) {
            Array(freqBins) { ComplexFloat(0f, 0f) }
        }
        val magnitudes = Array(numSensors) { FloatArray(freqBins) }
        val weightedMagnitudes = Array(numSensors) { FloatArray(freqBins) }

        for (sensor in 0 until numSensors) {
            for (bin in 0 until freqBins) {
                val real = outputFloatBuffer.get()
                val imag = outputFloatBuffer.get()
                val magnitude = outputFloatBuffer.get()
                val weighted = outputFloatBuffer.get()

                complexSpectrum[sensor][bin] = ComplexFloat(real, imag)
                magnitudes[sensor][bin] = magnitude
                weightedMagnitudes[sensor][bin] = weighted
            }
        }

        val fftResult = FftResult(
            complexSpectrum = complexSpectrum,
            magnitudes = magnitudes,
            weightedMagnitudes = weightedMagnitudes
        )

        return InferenceResult(
            output = fftResult,
            timing = InferenceTiming(
                transferMs = transferDuration,
                computeMs = computeDuration
            )
        )
    }

    fun close() {
        interpreter.close()
        (gpuDelegate as? GpuDelegate)?.close()
        (nnapiDelegate as? NnApiDelegate)?.close()
    }

    private fun nanosToMillis(value: Long): Double = value / 1_000_000.0

    companion object {
        private const val FLOAT_BYTES = 4
        private const val OUTPUT_FIELDS = 4
    }
}
