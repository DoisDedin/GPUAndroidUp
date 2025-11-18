package com.example.vulkanfft.logging

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ResultLogger {

    private val formatter = SimpleDateFormat("yyyy-MM-dd HH:mm:ss.SSS", Locale.US)

    fun append(context: Context, fileName: String, lines: List<String>) {
        val dir = logsDir(context)
        if (!dir.exists()) {
            dir.mkdirs()
        }
        val file = File(dir, fileName)
        val timestamp = formatter.format(Date())
        val builder = StringBuilder()
        builder.appendLine("===== $timestamp =====")
        lines.forEach { builder.appendLine(it) }
        builder.appendLine()
        file.appendText(builder.toString())
    }

    fun clearAll(context: Context): Boolean {
        val dir = logsDir(context)
        if (!dir.exists()) return true
        var success = true
        dir.listFiles()?.forEach { success = success && it.delete() }
        return success
    }

    fun buildShareableReport(context: Context): File? {
        val dir = logsDir(context)
        val files = dir.listFiles()
            ?.filter { it.isFile && it.name != "benchmarks_report.txt" }
            ?.sortedBy { it.name }
            ?: return null
        if (files.isEmpty()) return null
        val report = File(dir, "benchmarks_report.txt")
        val builder = StringBuilder()
        files.forEach { file ->
            builder.appendLine("===== ${file.name} =====")
            builder.appendLine(file.readText())
        }
        report.writeText(builder.toString())
        return report
    }

    fun logsDir(context: Context): File {
        return context.getExternalFilesDir("benchmarks") ?: File(context.filesDir, "benchmarks")
    }
}
