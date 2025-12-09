package com.example.vulkanfft.logging

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class BenchmarkEntry(
    val timestamp: Long,
    val testName: String,
    val processingMode: String,
    val delegate: String,
    val dataDescription: String,
    val inputSize: Int,
    val durationMs: Double,
    val durationStdDev: Double,
    val durationMin: Double,
    val durationMax: Double,
    val transferDurationMs: Double,
    val transferStdDev: Double,
    val transferMin: Double,
    val transferMax: Double,
    val computeDurationMs: Double,
    val computeStdDev: Double,
    val computeMin: Double,
    val computeMax: Double,
    val throughput: Double,
    val iterations: Int,
    val batchSize: Int,
    val estimatedEnergyImpact: String,
    val deviceInfo: DeviceInfo,
    val extraNotes: String,
    val batteryTempStartC: Double?,
    val batteryTempEndC: Double?,
    val cpuTempStartC: Double?,
    val cpuTempEndC: Double?,
    val gpuTempStartC: Double?,
    val gpuTempEndC: Double?
) {
    fun toCsvRow(): String {
        return listOf(
            timestamp.toString(),
            testName,
            processingMode,
            delegate,
            dataDescription,
            inputSize.toString(),
            String.format(Locale.US, "%.6f", durationMs),
            String.format(Locale.US, "%.6f", durationStdDev),
            String.format(Locale.US, "%.6f", durationMin),
            String.format(Locale.US, "%.6f", durationMax),
            String.format(Locale.US, "%.6f", transferDurationMs),
            String.format(Locale.US, "%.6f", transferStdDev),
            String.format(Locale.US, "%.6f", transferMin),
            String.format(Locale.US, "%.6f", transferMax),
            String.format(Locale.US, "%.6f", computeDurationMs),
            String.format(Locale.US, "%.6f", computeStdDev),
            String.format(Locale.US, "%.6f", computeMin),
            String.format(Locale.US, "%.6f", computeMax),
            String.format(Locale.US, "%.3f", throughput),
            iterations.toString(),
            batchSize.toString(),
            estimatedEnergyImpact,
            deviceInfo.manufacturer,
            deviceInfo.model,
            deviceInfo.hardware,
            deviceInfo.board,
            deviceInfo.soc ?: "",
            deviceInfo.sdkInt.toString(),
            deviceInfo.isPowerSaveMode.toString(),
            extraNotes.replace("\n", " "),
            batteryTempStartC?.let { "%.1f".format(Locale.US, it) } ?: "",
            batteryTempEndC?.let { "%.1f".format(Locale.US, it) } ?: "",
            cpuTempStartC?.let { "%.1f".format(Locale.US, it) } ?: "",
            cpuTempEndC?.let { "%.1f".format(Locale.US, it) } ?: "",
            gpuTempStartC?.let { "%.1f".format(Locale.US, it) } ?: "",
            gpuTempEndC?.let { "%.1f".format(Locale.US, it) } ?: ""
        ).joinToString(",") { value ->
            if (value.contains(",") || value.contains(" ")) "\"$value\"" else value
        }
    }

    fun toTextBlock(): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        val builder = StringBuilder()
        builder.appendLine("===== ${formatter.format(Date(timestamp))} =====")
        builder.appendLine("Teste: $testName")
        builder.appendLine("Modo: $processingMode | Delegate: $delegate")
        builder.appendLine("Dados: $dataDescription ($inputSize amostras)")
        builder.appendLine("Iterações: $iterations | Pacotes/batch: $batchSize")
        builder.appendLine(
            "Total: ${"%.3f".format(durationMs)}±${"%.3f".format(durationStdDev)} ms " +
                "(min=${"%.3f".format(durationMin)} / max=${"%.3f".format(durationMax)})"
        )
        builder.appendLine(
            "Transferência: ${"%.3f".format(transferDurationMs)} ms " +
                " | Processamento: ${"%.3f".format(computeDurationMs)} ms"
        )
        builder.appendLine("Throughput: ${"%.3f".format(throughput)} ops/s")
        builder.appendLine("Impacto energético estimado: $estimatedEnergyImpact")
        builder.appendLine("Dispositivo: ${deviceInfo.manufacturer} ${deviceInfo.model} (HW=${deviceInfo.hardware}, Board=${deviceInfo.board}, SDK=${deviceInfo.sdkInt})")
        builder.appendLine("Power saver ativo: ${deviceInfo.isPowerSaveMode}")
        if (
            batteryTempStartC != null || batteryTempEndC != null ||
            cpuTempStartC != null || cpuTempEndC != null ||
            gpuTempStartC != null || gpuTempEndC != null
        ) {
            builder.appendLine(
                "Temp Bateria: ${formatTemp(batteryTempStartC)} → ${formatTemp(batteryTempEndC)} | " +
                    "CPU: ${formatTemp(cpuTempStartC)} → ${formatTemp(cpuTempEndC)} | " +
                    "GPU: ${formatTemp(gpuTempStartC)} → ${formatTemp(gpuTempEndC)}"
            )
        }
        if (deviceInfo.soc != null) {
            builder.appendLine("SoC: ${deviceInfo.soc}")
        }
        builder.appendLine("Notas: $extraNotes")
        builder.appendLine()
        return builder.toString()
    }

    private fun formatTemp(value: Double?): String =
        value?.let { "%.1f°C".format(it) } ?: "-"
}

object BenchmarkReporter {

    private const val CSV_NAME = "benchmark_results.csv"
    private const val TXT_NAME = "benchmark_results.txt"
    private val csvHeader = listOf(
        "timestamp",
        "test_name",
        "processing_mode",
        "delegate",
        "data_description",
        "input_size",
        "duration_ms",
        "duration_std_ms",
        "duration_min_ms",
        "duration_max_ms",
        "transfer_ms",
        "transfer_std_ms",
        "transfer_min_ms",
        "transfer_max_ms",
        "compute_ms",
        "compute_std_ms",
        "compute_min_ms",
        "compute_max_ms",
        "throughput_ops_per_sec",
        "iterations",
        "batch_size",
        "estimated_energy",
        "manufacturer",
        "model",
        "hardware",
        "board",
        "soc",
        "sdk_int",
        "power_save",
        "notes",
        "battery_temp_start_c",
        "battery_temp_end_c",
        "cpu_temp_start_c",
        "cpu_temp_end_c",
        "gpu_temp_start_c",
        "gpu_temp_end_c"
    ).joinToString(",")

    fun append(context: Context, entry: BenchmarkEntry) {
        val dir = ResultLogger.logsDir(context)
        if (!dir.exists()) dir.mkdirs()

        val csvFile = File(dir, CSV_NAME)
        if (!csvFile.exists()) {
            csvFile.writeText(csvHeader + "\n")
        }
        csvFile.appendText(entry.toCsvRow() + "\n")

        val txtFile = File(dir, TXT_NAME)
        txtFile.appendText(entry.toTextBlock())
    }
}
