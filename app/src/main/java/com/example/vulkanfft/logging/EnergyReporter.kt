package com.example.vulkanfft.logging

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class EnergyLogEntry(
    val timestamp: Long,
    val scenarioLabel: String,
    val delegate: String,
    val batchMode: String,
    val durationMinutes: Int,
    val totalRuns: Int,
    val intervalSeconds: Int,
    val batteryStartPercent: Double,
    val batteryEndPercent: Double,
    val batteryDropPercent: Double,
    val batteryStartChargeMah: Double?,
    val batteryEndChargeMah: Double?,
    val energyStartNwh: Double?,
    val energyEndNwh: Double?,
    val temperatureStartC: Double?,
    val temperatureEndC: Double?,
    val isChargingStart: Boolean,
    val isChargingEnd: Boolean,
    val powerSaveStart: Boolean,
    val powerSaveEnd: Boolean,
    val notes: String
) {
    fun toCsvRow(): String {
        return listOf(
            timestamp.toString(),
            scenarioLabel,
            delegate,
            batchMode,
            durationMinutes.toString(),
            totalRuns.toString(),
            intervalSeconds.toString(),
            "%.2f".format(Locale.US, batteryStartPercent),
            "%.2f".format(Locale.US, batteryEndPercent),
            "%.2f".format(Locale.US, batteryDropPercent),
            batteryStartChargeMah?.let { "%.3f".format(Locale.US, it) } ?: "",
            batteryEndChargeMah?.let { "%.3f".format(Locale.US, it) } ?: "",
            energyStartNwh?.let { "%.3f".format(Locale.US, it) } ?: "",
            energyEndNwh?.let { "%.3f".format(Locale.US, it) } ?: "",
            temperatureStartC?.let { "%.1f".format(Locale.US, it) } ?: "",
            temperatureEndC?.let { "%.1f".format(Locale.US, it) } ?: "",
            isChargingStart.toString(),
            isChargingEnd.toString(),
            powerSaveStart.toString(),
            powerSaveEnd.toString(),
            notes.replace("\n", " ")
        ).joinToString(",") { value ->
            if (value.contains(",") || value.contains(" ")) "\"$value\"" else value
        }
    }

    fun toTextBlock(): String {
        val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.getDefault())
        return buildString {
            appendLine("===== ${formatter.format(Date(timestamp))} =====")
            appendLine("Cenário: $scenarioLabel")
            appendLine("Delegate: $delegate | Batch: $batchMode")
            appendLine("Duração: $durationMinutes min ($totalRuns execuções a cada $intervalSeconds s)")
            appendLine(
                "Bateria: ${"%.2f".format(batteryStartPercent)}% → " +
                    "${"%.2f".format(batteryEndPercent)}% (Δ ${"%.2f".format(batteryDropPercent)}%)"
            )
            appendLine(
                "Temperatura: ${formatNullable(temperatureStartC)}°C → " +
                    "${formatNullable(temperatureEndC)}°C"
            )
            appendLine("Power saver: $powerSaveStart → $powerSaveEnd | Carregando: $isChargingStart → $isChargingEnd")
            appendLine("Notas: $notes")
            appendLine()
        }
    }

    private fun formatNullable(value: Double?): String =
        value?.let { "%.1f".format(Locale.US, it) } ?: "-"
}

object EnergyReporter {
    private const val CSV_NAME = "energy_results.csv"
    private const val TXT_NAME = "energy_results.txt"
    private val csvHeader = listOf(
        "timestamp",
        "scenario",
        "delegate",
        "batch_mode",
        "duration_minutes",
        "total_runs",
        "interval_seconds",
        "battery_start_percent",
        "battery_end_percent",
        "battery_drop_percent",
        "battery_start_mah",
        "battery_end_mah",
        "energy_start_nwh",
        "energy_end_nwh",
        "temperature_start_c",
        "temperature_end_c",
        "is_charging_start",
        "is_charging_end",
        "power_save_start",
        "power_save_end",
        "notes"
    ).joinToString(",")

    fun append(context: Context, entry: EnergyLogEntry) {
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
