package com.example.vulkanfft

import android.annotation.SuppressLint
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlin.math.roundToInt

class EnergyTestService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val benchmarkExecutor = BenchmarkExecutor()
    private var wakeLock: PowerManager.WakeLock? = null
    private var runningJob: Job? = null
    private var configuredMadScale: DataScale = DataScale.BASE
    private var configuredFftScale: DataScale = DataScale.BASE

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return when (intent?.action) {
            ACTION_START -> {
                configuredMadScale = intent.getStringExtra(EXTRA_MAD_SCALE).toDataScale(DataScale.BASE)
                configuredFftScale = intent.getStringExtra(EXTRA_FFT_SCALE).toDataScale(DataScale.BASE)
                startEnergyTests()
                START_STICKY
            }
            ACTION_STOP -> {
                stopEnergyTests()
                START_NOT_STICKY
            }
            else -> {
                stopSelf()
                START_NOT_STICKY
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun startEnergyTests() {
        if (runningJob?.isActive == true) return
        val notification = buildNotification(getString(R.string.energy_test_notification_desc))
        startForeground(NOTIFICATION_ID, notification)
        wakeLock = acquireWakeLock()
        runningJob = serviceScope.launch {
            try {
                EnergyTestBus.emit(EnergyTestEvent.Started)
                val summary = benchmarkExecutor.runEnergyTests(
                    context = applicationContext,
                    iterations = BenchmarkExecutor.ENERGY_TEST_ITERATIONS,
                    madScale = configuredMadScale,
                    fftScale = configuredFftScale,
                    onScenarioStart = { current, total, label, scale ->
                        val percent = calculatePercent(current - 1, total)
                        val statusText = formatStatusText(current, total, percent, scale, label)
                        updateNotification(statusText)
                        EnergyTestBus.tryEmit(
                            EnergyTestEvent.Status(
                                current = current,
                                total = total,
                                scenarioLabel = label,
                                percentComplete = percent,
                                scale = scale
                            )
                        )
                    },
                    onScenarioComplete = { snapshot ->
                        val progressText = formatCompletionText(snapshot)
                        updateNotification(progressText)
                        EnergyTestBus.tryEmit(EnergyTestEvent.Progress(snapshot))
                    }
                )
                EnergyTestBus.emit(EnergyTestEvent.Finished(summary))
            } catch (ex: CancellationException) {
                EnergyTestBus.tryEmit(EnergyTestEvent.Cancelled)
            } catch (ex: Exception) {
                EnergyTestBus.tryEmit(
                    EnergyTestEvent.Error(ex.message ?: "Erro inesperado durante o teste energético.")
                )
            } finally {
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
            }
        }
    }

    private fun stopEnergyTests() {
        runningJob?.cancel()
    }

    private fun updateNotification(text: String) {
        val notification = buildNotification(text)
        val manager = getSystemService(NotificationManager::class.java)
        manager.notify(NOTIFICATION_ID, notification)
    }

    private fun calculatePercent(count: Int, total: Int): Int {
        if (total <= 0) return 0
        return ((count.toDouble() / total) * 100).roundToInt().coerceIn(0, 100)
    }

    private fun formatStatusText(
        current: Int,
        total: Int,
        percent: Int,
        scale: DataScale,
        label: String
    ): String {
        return "Executando $current/$total (${percent}% · ${scale.shortLabel}) - $label"
    }

    private fun formatCompletionText(snapshot: BenchmarkExecutor.EnergyScenarioSnapshot): String {
        val base =
            "Concluído ${snapshot.scenarioIndex}/${snapshot.totalScenarios} (${snapshot.percentComplete}% · ${snapshot.scale.shortLabel}) - ${snapshot.scenario.displayLabel}"
        val temp = snapshot.temperatureSummary ?: formatTemperatureSummary(
            snapshot.temperatureStartC,
            snapshot.temperatureEndC,
            snapshot.cpuTemperatureStartC,
            snapshot.cpuTemperatureEndC,
            snapshot.gpuTemperatureStartC,
            snapshot.gpuTemperatureEndC
        )
        return if (temp != null) "$base | $temp" else base
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val manager = getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.energy_test_notification_title),
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description = getString(R.string.energy_test_notification_desc)
        }
        manager.createNotificationChannel(channel)
    }

    private fun buildNotification(contentText: String): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(getString(R.string.energy_test_notification_title))
            .setContentText(contentText)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    @SuppressLint("WakelockTimeout")
    private fun acquireWakeLock(): PowerManager.WakeLock? {
        val pm = getSystemService(PowerManager::class.java) ?: return null
        return try {
            pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG).apply { acquire() }
        } catch (_: SecurityException) {
            null
        } catch (_: RuntimeException) {
            null
        }
    }

    private fun releaseWakeLock() {
        try {
            if (wakeLock?.isHeld == true) {
                wakeLock?.release()
            }
        } catch (_: RuntimeException) {
        }
        wakeLock = null
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        releaseWakeLock()
    }

    private fun String?.toDataScale(default: DataScale): DataScale {
        if (this.isNullOrBlank()) return default
        return runCatching { DataScale.valueOf(this) }.getOrDefault(default)
    }

    private fun formatTemperatureSummary(
        batteryStart: Double?,
        batteryEnd: Double?,
        cpuStart: Double?,
        cpuEnd: Double?,
        gpuStart: Double?,
        gpuEnd: Double?
    ): String? {
        val segments = buildList {
            if (batteryStart != null || batteryEnd != null) {
                add("Bateria ${formatTemp(batteryStart)} -> ${formatTemp(batteryEnd)}")
            }
            if (cpuStart != null || cpuEnd != null) {
                add("CPU ${formatTemp(cpuStart)} -> ${formatTemp(cpuEnd)}")
            }
            if (gpuStart != null || gpuEnd != null) {
                add("GPU ${formatTemp(gpuStart)} -> ${formatTemp(gpuEnd)}")
            }
        }
        if (segments.isEmpty()) return null
        return "Temp ${segments.joinToString(" | ")}"
    }

    private fun formatTemp(value: Double?): String = value?.let { "%.1f°C".format(it) } ?: "-"

    companion object {
        private const val CHANNEL_ID = "energy_tests_channel"
        private const val NOTIFICATION_ID = 1001
        private const val ACTION_START = "com.example.vulkanfft.energy.START"
        private const val ACTION_STOP = "com.example.vulkanfft.energy.STOP"
        private const val WAKE_LOCK_TAG = "VulkanFFT:EnergyTest"
        private const val EXTRA_MAD_SCALE = "com.example.vulkanfft.extra.MAD_SCALE"
        private const val EXTRA_FFT_SCALE = "com.example.vulkanfft.extra.FFT_SCALE"

        fun start(context: Context, madScale: DataScale, fftScale: DataScale) {
            val intent = Intent(context, EnergyTestService::class.java).apply {
                action = ACTION_START
                putExtra(EXTRA_MAD_SCALE, madScale.name)
                putExtra(EXTRA_FFT_SCALE, fftScale.name)
            }
            ContextCompat.startForegroundService(context, intent)
        }

        fun stop(context: Context) {
            val intent = Intent(context, EnergyTestService::class.java).apply {
                action = ACTION_STOP
            }
            ContextCompat.startForegroundService(context, intent)
        }
    }
}
