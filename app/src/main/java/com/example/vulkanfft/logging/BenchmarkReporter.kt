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
    val throughput: Double,
    val estimatedEnergyImpact: String,
    val deviceInfo: DeviceInfo,
    val extraNotes: String
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
            String.format(Locale.US, "%.3f", throughput),
            estimatedEnergyImpact,
            deviceInfo.manufacturer,
            deviceInfo.model,
            deviceInfo.hardware,
            deviceInfo.board,
            deviceInfo.soc ?: "",
            deviceInfo.sdkInt.toString(),
            deviceInfo.isPowerSaveMode.toString(),
            extraNotes.replace("\n", " ")
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
        builder.appendLine("Duração: ${"%.3f".format(durationMs)} ms | Throughput: ${"%.3f".format(throughput)} ops/s")
        builder.appendLine("Impacto energético estimado: $estimatedEnergyImpact")
        builder.appendLine("Dispositivo: ${deviceInfo.manufacturer} ${deviceInfo.model} (HW=${deviceInfo.hardware}, Board=${deviceInfo.board}, SDK=${deviceInfo.sdkInt})")
        builder.appendLine("Power saver ativo: ${deviceInfo.isPowerSaveMode}")
        if (deviceInfo.soc != null) {
            builder.appendLine("SoC: ${deviceInfo.soc}")
        }
        builder.appendLine("Notas: $extraNotes")
        builder.appendLine()
        return builder.toString()
    }
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
        "throughput_ops_per_sec",
        "estimated_energy",
        "manufacturer",
        "model",
        "hardware",
        "board",
        "soc",
        "sdk_int",
        "power_save",
        "notes"
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
