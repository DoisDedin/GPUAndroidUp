package com.example.vulkanfft

import android.content.Context
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.vulkanfft.util.DelegateType
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class FirstViewModel : ViewModel() {

    private val benchmarkExecutor = BenchmarkExecutor()
    private val energyLog = mutableListOf<String>()
    private var currentEnergyStatus: String? = null
    private var benchmarkIterations = DEFAULT_ITERATIONS_OPTION
    private var benchmarkBatchSize = DEFAULT_BATCH_OPTION

    private val _benchmarkResult = MutableLiveData("Aguardando execuções.")
    val benchmarkResult: LiveData<String> = _benchmarkResult

    private val _energyResult = MutableLiveData("Sem testes energéticos recentes.")
    val energyResult: LiveData<String> = _energyResult

    private val _progress = MutableLiveData<BenchmarkProgress>()
    val progress: LiveData<BenchmarkProgress> = _progress

    private val _energyInProgress = MutableLiveData(false)
    val energyInProgress: LiveData<Boolean> = _energyInProgress

    private val _sessionRunCount = MutableLiveData(0)
    val sessionRunCount: LiveData<Int> = _sessionRunCount

    init {
        observeEnergyEvents()
    }

    private fun observeEnergyEvents() {
        viewModelScope.launch {
            EnergyTestBus.events.collectLatest { event ->
                when (event) {
                    EnergyTestEvent.Started -> {
                        energyLog.clear()
                        currentEnergyStatus = "Preparando testes energéticos..."
                        _energyInProgress.postValue(true)
                        publishEnergyDisplay()
                    }

                    is EnergyTestEvent.Status -> {
                        currentEnergyStatus =
                            "Executando ${event.current}/${event.total} (${event.percentComplete}% · ${event.scale.shortLabel}) - ${event.scenarioLabel}"
                        publishEnergyDisplay()
                    }

                    is EnergyTestEvent.Progress -> {
                        val message =
                            "[${event.snapshot.percentComplete}% · ${event.snapshot.scale.shortLabel}] ${event.snapshot.summaryLine}"
                        energyLog += message
                        publishEnergyDisplay()
                    }

                    is EnergyTestEvent.Finished -> {
                        currentEnergyStatus = null
                        energyLog.clear()
                        _energyResult.postValue(event.summary)
                        _energyInProgress.postValue(false)
                    }

                    is EnergyTestEvent.Error -> {
                        currentEnergyStatus = null
                        energyLog.clear()
                        _energyResult.postValue("Erro: ${event.message}")
                        _energyInProgress.postValue(false)
                    }

                    EnergyTestEvent.Cancelled -> {
                        currentEnergyStatus = null
                        energyLog.clear()
                        _energyResult.postValue("Teste energético cancelado.")
                        _energyInProgress.postValue(false)
                    }
                }
            }
        }
    }

    private fun publishEnergyDisplay() {
        val status = currentEnergyStatus ?: "Testes energéticos em execução..."
        val text = if (energyLog.isEmpty()) {
            status
        } else {
            buildString {
                appendLine(status)
                energyLog.forEach { appendLine(it) }
            }.trim()
        }
        _energyResult.postValue(text)
    }

    fun runScenario(
        context: Context,
        scenario: BenchmarkScenario,
        scale: DataScale = DataScale.BASE
    ) {
        viewModelScope.launch(Dispatchers.Default) {
            val iterations = benchmarkIterations
            val batchSize = benchmarkBatchSize
            val summary = benchmarkExecutor.runScenario(
                context = context.applicationContext,
                scenario = scenario,
                iterations = iterations,
                scale = scale,
                batchSizeOverride = batchSize
            )
            _benchmarkResult.postValue(summary.summaryText)
            incrementSessionCounter()
        }
    }

    fun runFullBenchmarkSuite(context: Context) {
        runSuiteAcrossScales(
            context = context,
            scales = DataScale.unifiedScales,
            startMessage = "Iniciando suíte completa (512 → 526k)",
            completionMessage = "Suíte completa finalizada"
        )
    }

    private fun runSuiteAcrossScales(
        context: Context,
        scales: List<DataScale>,
        startMessage: String,
        completionMessage: String
    ) {
        if (scales.isEmpty()) {
            _benchmarkResult.postValue("Nenhuma escala disponível para execução.")
            return
        }
        viewModelScope.launch(Dispatchers.Default) {
            val ctx = context.applicationContext
            val scenarios = BenchmarkScenario.suiteOrder
            val iterations = benchmarkIterations
            val batchSize = benchmarkBatchSize
            val totalSteps = scenarios.size * scales.size
            var currentStep = 0
            _progress.postValue(
                BenchmarkProgress(
                    total = totalSteps,
                    current = 0,
                    running = true,
                    message = startMessage
                )
            )
            scales.forEach { scale ->
                scenarios.forEach { scenario ->
                    val summary = benchmarkExecutor.runScenario(
                        context = ctx,
                        scenario = scenario,
                        iterations = iterations,
                        scale = scale,
                        batchSizeOverride = batchSize
                    )
                    currentStep++
                    _benchmarkResult.postValue(summary.summaryText)
                    incrementSessionCounter()
                    _progress.postValue(
                        BenchmarkProgress(
                            total = totalSteps,
                            current = currentStep,
                            running = currentStep < totalSteps,
                            message = "Executado: ${scenario.displayLabel} (${scale.shortLabel})"
                        )
                    )
                }
            }
            _progress.postValue(
                BenchmarkProgress(
                    total = totalSteps,
                    current = totalSteps,
                    running = false,
                    message = completionMessage
                )
            )
        }
    }

    fun startEnergyTest(
        context: Context,
        madScale: DataScale,
        fftScale: DataScale
    ) {
        if (_energyInProgress.value == true) {
            _energyResult.postValue("Teste energético já está em execução.")
            return
        }
        EnergyTestService.start(context.applicationContext, madScale, fftScale)
    }

    fun cancelEnergyTest(context: Context) {
        EnergyTestService.stop(context.applicationContext)
    }

    fun updateBenchmarkIterations(value: Int) {
        benchmarkIterations = value.coerceAtLeast(1)
    }

    fun currentBenchmarkIterations(): Int = benchmarkIterations

    fun updateBatchSize(value: Int) {
        benchmarkBatchSize = value.coerceAtLeast(1)
    }

    fun currentBatchSize(): Int = benchmarkBatchSize

    fun currentSessionCount(): Int = _sessionRunCount.value ?: 0

    fun resetSessionCounter() {
        _sessionRunCount.postValue(0)
    }

    private fun incrementSessionCounter(delta: Int = 1) {
        val current = _sessionRunCount.value ?: 0
        _sessionRunCount.postValue(current + delta.coerceAtLeast(0))
    }
}

data class BenchmarkProgress(
    val total: Int,
    val current: Int,
    val running: Boolean,
    val message: String
)

data class AccelerometerSample(
    val timestamp: Long,
    val x: Int,
    val y: Int,
    val z: Int
)

data class ScenarioRunSummary(
    val scenario: BenchmarkScenario,
    val summaryText: String,
    val timings: TimingStats,
    val notes: String,
    val batchSize: Int,
    val scale: DataScale
)

data class StatsSummary(
    val mean: Double,
    val stdDev: Double,
    val min: Double,
    val max: Double
) {
    companion object {
        val ZERO = StatsSummary(0.0, 0.0, 0.0, 0.0)
    }
}

data class TimingStats(
    val total: StatsSummary,
    val transfer: StatsSummary,
    val compute: StatsSummary
)

enum class DataScale(
    val shortLabel: String,
    val madVectorLength: Int,
    val fftSignalLength: Int,
    val isDefault: Boolean
) {
    BASE("1x", 4096, 4096, true),
    DOUBLE("2x", 8192, 8192, true),
    QUADRUPLE("4x", 16384, 16384, true),
    MICRO_512("512", 512, 512, false),
    MICRO_1K("1k", 1024, 1024, false),
    MICRO_2K("2k", 2048, 2048, false),
    EXTREME_32K("32k", 32_768, 32_768, false),
    EXTREME_64K("64k", 65_536, 65_536, false),
    EXTREME_128K("128k", 131_072, 131_072, false),
    EXTREME_256K("256k", 262_144, 262_144, false),
    EXTREME_526K("526k", 526_000, 526_000, false);

    companion object {
        val defaultScales: List<DataScale> = listOf(BASE, DOUBLE, QUADRUPLE)
        val focusedScales: List<DataScale> = listOf(MICRO_512, MICRO_1K, MICRO_2K, EXTREME_128K)
        val extremeScales: List<DataScale> = listOf(EXTREME_32K, EXTREME_64K, EXTREME_128K)
        val experimentalScales: List<DataScale> = listOf(EXTREME_256K, EXTREME_526K)
        val unifiedScales: List<DataScale> = listOf(
            MICRO_512,
            MICRO_1K,
            MICRO_2K,
            BASE,
            DOUBLE,
            QUADRUPLE,
            EXTREME_32K,
            EXTREME_64K
        )
    }
}

enum class Algorithm { MAD, FFT }

enum class BatchMode(val displayName: String) {
    SINGLE("1 pacote"),
    TEN_PACKETS("10 pacotes")
}

enum class DelegateMode(
    val displayName: String,
    val delegateType: DelegateType?
) {
    CPU_NATIVE("CPU Kotlin", null),
    TFLITE_CPU("TFLite CPU", DelegateType.CPU),
    TFLITE_GPU("TFLite GPU", DelegateType.GPU),
    TFLITE_NNAPI("TFLite NNAPI", DelegateType.NNAPI)
}

enum class BenchmarkScenario(
    val displayLabel: String,
    val algorithm: Algorithm,
    val delegateMode: DelegateMode,
    val batchMode: BatchMode,
    val logFileName: String,
    val processingMode: String
) {
    MAD_CPU_SINGLE(
        "MAD CPU Kotlin",
        Algorithm.MAD,
        DelegateMode.CPU_NATIVE,
        BatchMode.SINGLE,
        "mad_cpu_single.txt",
        "Kotlin"
    ),
    MAD_TFLITE_CPU_SINGLE(
        "MAD TFLite CPU",
        Algorithm.MAD,
        DelegateMode.TFLITE_CPU,
        BatchMode.SINGLE,
        "mad_tflite_cpu_single.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_GPU_SINGLE(
        "MAD TFLite GPU",
        Algorithm.MAD,
        DelegateMode.TFLITE_GPU,
        BatchMode.SINGLE,
        "mad_tflite_gpu_single.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_NN_SINGLE(
        "MAD TFLite NNAPI",
        Algorithm.MAD,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.SINGLE,
        "mad_tflite_nn_single.txt",
        "TensorFlow Lite"
    ),
    MAD_CPU_BATCH(
        "MAD CPU Kotlin x10",
        Algorithm.MAD,
        DelegateMode.CPU_NATIVE,
        BatchMode.TEN_PACKETS,
        "mad_cpu_batch.txt",
        "Kotlin"
    ),
    MAD_TFLITE_CPU_BATCH(
        "MAD TFLite CPU x10",
        Algorithm.MAD,
        DelegateMode.TFLITE_CPU,
        BatchMode.TEN_PACKETS,
        "mad_tflite_cpu_batch.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_GPU_BATCH(
        "MAD TFLite GPU x10",
        Algorithm.MAD,
        DelegateMode.TFLITE_GPU,
        BatchMode.TEN_PACKETS,
        "mad_tflite_gpu_batch.txt",
        "TensorFlow Lite"
    ),
    MAD_TFLITE_NN_BATCH(
        "MAD TFLite NNAPI x10",
        Algorithm.MAD,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.TEN_PACKETS,
        "mad_tflite_nn_batch.txt",
        "TensorFlow Lite"
    ),
    FFT_CPU_SINGLE(
        "FFT CPU",
        Algorithm.FFT,
        DelegateMode.CPU_NATIVE,
        BatchMode.SINGLE,
        "fft_cpu_single.txt",
        "Kotlin"
    ),
    FFT_TFLITE_CPU_SINGLE(
        "FFT TFLite CPU",
        Algorithm.FFT,
        DelegateMode.TFLITE_CPU,
        BatchMode.SINGLE,
        "fft_tflite_cpu_single.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_GPU_SINGLE(
        "FFT TFLite GPU",
        Algorithm.FFT,
        DelegateMode.TFLITE_GPU,
        BatchMode.SINGLE,
        "fft_tflite_gpu_single.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_NN_SINGLE(
        "FFT TFLite NNAPI",
        Algorithm.FFT,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.SINGLE,
        "fft_tflite_nn_single.txt",
        "TensorFlow Lite"
    ),
    FFT_CPU_BATCH(
        "FFT CPU x10",
        Algorithm.FFT,
        DelegateMode.CPU_NATIVE,
        BatchMode.TEN_PACKETS,
        "fft_cpu_batch.txt",
        "Kotlin"
    ),
    FFT_TFLITE_CPU_BATCH(
        "FFT TFLite CPU x10",
        Algorithm.FFT,
        DelegateMode.TFLITE_CPU,
        BatchMode.TEN_PACKETS,
        "fft_tflite_cpu_batch.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_GPU_BATCH(
        "FFT TFLite GPU x10",
        Algorithm.FFT,
        DelegateMode.TFLITE_GPU,
        BatchMode.TEN_PACKETS,
        "fft_tflite_gpu_batch.txt",
        "TensorFlow Lite"
    ),
    FFT_TFLITE_NN_BATCH(
        "FFT TFLite NNAPI x10",
        Algorithm.FFT,
        DelegateMode.TFLITE_NNAPI,
        BatchMode.TEN_PACKETS,
        "fft_tflite_nn_batch.txt",
        "TensorFlow Lite"
    );

    companion object {
        val suiteOrder = values().toList()
    }
}

private const val DEFAULT_ITERATIONS_OPTION = 4
private const val DEFAULT_BATCH_OPTION = 4
