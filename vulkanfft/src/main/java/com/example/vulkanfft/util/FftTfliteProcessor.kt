package com.example.vulkanfft.util

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Delegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.channels.FileChannel

class FftTfliteProcessor(
    context: Context,
    delegateType: DelegateType,
    private val numSensors: Int = 10,
    private val signalLength: Int = 10
) {

    private val tag = "FftTfliteProcessor"
    private val freqBins = signalLength / 2 + 1
    private val interpreter: Interpreter
    private val gpuDelegate: Delegate?
    private val nnapiDelegate: Delegate?
    private val modelVariant: ModelVariant
    private val samplesInputIndex: Int
    private val weightsInputIndex: Int?
    private val inputTensorCount: Int
    private val weightsInputBuffer: ByteBuffer?
    private val weightsFloatBuffer: FloatBuffer?
    private val samplesInputBuffer: ByteBuffer
    private val samplesFloatBuffer: FloatBuffer
    private val outputBuffer: ByteBuffer
    private val outputFloatBuffer: FloatBuffer
    private val outputs: MutableMap<Int, Any>
    private val inputArray: Array<Any>

    init {
        val options = Interpreter.Options()
        var localGpuDelegate: Delegate? = null
        var localNnapiDelegate: Delegate? = null

        when (delegateType) {
            DelegateType.GPU -> {
                val gpuOptions = GpuDelegate.Options().apply {
                    setPrecisionLossAllowed(false)
                    setInferencePreference(GpuDelegate.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                }
                localGpuDelegate = GpuDelegate(gpuOptions).also { options.addDelegate(it) }
                Log.d(tag, "Usando delegate GPU (full precision)")
            }

            DelegateType.NNAPI -> {
                localNnapiDelegate = NnApiDelegate().also { options.addDelegate(it) }
                Log.d(tag, "Usando delegate NNAPI")
            }

            DelegateType.CPU -> {
                options.setUseXNNPACK(true)
                Log.d(tag, "Usando delegate CPU (XNNPACK)")
            }
        }
        gpuDelegate = localGpuDelegate
        nnapiDelegate = localNnapiDelegate

        val modelBuffer = loadModelFile(context, signalLength)
        val createdInterpreter = Interpreter(modelBuffer, options)
        createdInterpreter.allocateTensors()

        inputTensorCount = createdInterpreter.inputTensorCount
        require(inputTensorCount in 1..2) {
            "Modelo FFT TFLite com quantidade inválida de inputs ($inputTensorCount)"
        }

        if (inputTensorCount == 2) {
            val shape0 = createdInterpreter.getInputTensor(0).shape()
            val shape1 = createdInterpreter.getInputTensor(1).shape()
            val (samplesIdx, weightsIdx) = when {
                shape0.lastOrNull() == signalLength && shape1.lastOrNull() == freqBins -> 0 to 1
                shape1.lastOrNull() == signalLength && shape0.lastOrNull() == freqBins -> 1 to 0
                shape0.lastOrNull() == signalLength -> 0 to 1
                shape1.lastOrNull() == signalLength -> 1 to 0
                else -> 0 to 1
            }
            samplesInputIndex = samplesIdx
            weightsInputIndex = weightsIdx
            modelVariant = ModelVariant.STACKED_OUTPUT_WITH_WEIGHTS
        } else {
            samplesInputIndex = 0
            weightsInputIndex = null
            modelVariant = ModelVariant.MAGNITUDE_ONLY
        }

        weightsInputBuffer = if (weightsInputIndex != null) {
            ByteBuffer.allocateDirect(numSensors * freqBins * FLOAT_BYTES).order(ByteOrder.nativeOrder())
        } else {
            null
        }
        weightsFloatBuffer = weightsInputBuffer?.asFloatBuffer()
        samplesInputBuffer =
            ByteBuffer.allocateDirect(numSensors * signalLength * FLOAT_BYTES).order(ByteOrder.nativeOrder())
        samplesFloatBuffer = samplesInputBuffer.asFloatBuffer()

        val outputFieldCount = if (modelVariant == ModelVariant.STACKED_OUTPUT_WITH_WEIGHTS) OUTPUT_FIELDS else 1
        outputBuffer = ByteBuffer.allocateDirect(numSensors * freqBins * outputFieldCount * FLOAT_BYTES)
            .order(ByteOrder.nativeOrder())
        outputFloatBuffer = outputBuffer.asFloatBuffer()

        createdInterpreter.resizeInput(samplesInputIndex, intArrayOf(numSensors, signalLength))
        weightsInputIndex?.let {
            createdInterpreter.resizeInput(it, intArrayOf(numSensors, freqBins))
        }
        createdInterpreter.allocateTensors()
        outputs = hashMapOf(0 to outputBuffer)
        interpreter = createdInterpreter

        inputArray = if (weightsInputIndex != null) {
            Array<Any>(inputTensorCount) { samplesInputBuffer }.apply {
                this[samplesInputIndex] = samplesInputBuffer
                this[weightsInputIndex!!] = weightsInputBuffer!!
            }
        } else {
            arrayOf(samplesInputBuffer as Any)
        }

        Log.d(
            tag,
            "FFT model variant: $modelVariant (inputs=$inputTensorCount, samplesIndex=$samplesInputIndex, weightsIndex=$weightsInputIndex)"
        )
    }

    private fun loadModelFile(context: Context, signalLength: Int): ByteBuffer {
        val assetManager = context.assets
        val candidate = "fft_model_${signalLength}.tflite"
        val descriptor = try {
            assetManager.openFd(candidate)
        } catch (_: IOException) {
            assetManager.openFd("fft_model.tflite")
        }
        val startOffset = descriptor.startOffset
        val declaredLength = descriptor.declaredLength
        val inputStream = FileInputStream(descriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val mapped = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        inputStream.close()
        descriptor.close()
        return mapped
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
        weightsFloatBuffer?.let { buffer ->
            buffer.rewind()
            for (sensor in 0 until numSensors) {
                val sensorWeights = weights[sensor]
                for (bin in 0 until freqBins) {
                    buffer.put(sensorWeights[bin])
                }
            }
            weightsInputBuffer?.rewind()
        }
        samplesFloatBuffer.rewind()
        for (sensor in 0 until numSensors) {
            val sensorSamples = samples[sensor]
            for (sample in 0 until signalLength) {
                samplesFloatBuffer.put(sensorSamples[sample])
            }
        }
        samplesInputBuffer.rewind()
        val transferDuration = nanosToMillis(System.nanoTime() - transferStart)

        val computeStart = System.nanoTime()
        interpreter.runForMultipleInputsOutputs(inputArray, outputs)
        val computeDuration = nanosToMillis(System.nanoTime() - computeStart)
        Log.d(tag, "Transferência FFT: ${"%.3f".format(transferDuration)} ms")
        Log.d(tag, "Processamento FFT: ${"%.3f".format(computeDuration)} ms")

        outputBuffer.rewind()
        outputFloatBuffer.rewind()

        val complexSpectrum = Array(numSensors) { Array(freqBins) { ComplexFloat(0f, 0f) } }
        val magnitudes = Array(numSensors) { FloatArray(freqBins) }
        val weightedMagnitudes = Array(numSensors) { FloatArray(freqBins) }

        when (modelVariant) {
            ModelVariant.STACKED_OUTPUT_WITH_WEIGHTS -> {
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
            }

            ModelVariant.MAGNITUDE_ONLY -> {
                for (sensor in 0 until numSensors) {
                    val currentWeights = weights[sensor]
                    for (bin in 0 until freqBins) {
                        val magnitude = outputFloatBuffer.get()
                        magnitudes[sensor][bin] = magnitude
                        weightedMagnitudes[sensor][bin] = magnitude * currentWeights[bin]
                    }
                }
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

    private enum class ModelVariant {
        STACKED_OUTPUT_WITH_WEIGHTS,
        MAGNITUDE_ONLY
    }
}
