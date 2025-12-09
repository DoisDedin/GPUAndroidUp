package com.example.vulkanfft

import android.content.Context
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class MadErrorMetrics(
    val relativeMean: Double,
    val relativeStd: Double,
    val relativeMax: Double,
    val absoluteMin: Double
)

object PrecisionReporter {

    private const val DIRECTORY = "precision"
    private const val FFT_CSV = "precision_fft.csv"
    private const val MAD_CSV = "precision_mad.csv"
    private val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)

    fun appendFft(
        context: Context,
        signalLength: Int,
        numSensors: Int,
        delegate: String,
        metrics: TfliteDelegatesInstrumentedTest.FftErrorMetrics
    ) {
        val file = ensureFile(context, FFT_CSV, listOf(
            "timestamp",
            "delegate",
            "signal_length",
            "num_sensors",
            "max_rel_diff",
            "mean_rel_diff",
            "max_abs_diff",
            "mean_abs_diff",
            "rmse"
        ))
        val row = listOf(
            dateFormat.format(Date()),
            delegate,
            signalLength.toString(),
            numSensors.toString(),
            metrics.maxRelativeDiff.toString(),
            metrics.meanRelativeDiff.toString(),
            metrics.maxAbsoluteDiff.toString(),
            metrics.meanAbsoluteDiff.toString(),
            metrics.rmse.toString()
        ).joinToString(",")
        file.appendText(row + "\n")
    }

    fun appendMad(
        context: Context,
        vectorLength: Int,
        metrics: MadErrorMetrics
    ) {
        val file = ensureFile(context, MAD_CSV, listOf(
            "timestamp",
            "vector_length",
            "relative_mean_diff",
            "relative_std_diff",
            "relative_max_diff",
            "absolute_min_diff"
        ))
        val row = listOf(
            dateFormat.format(Date()),
            vectorLength.toString(),
            metrics.relativeMean.toString(),
            metrics.relativeStd.toString(),
            metrics.relativeMax.toString(),
            metrics.absoluteMin.toString()
        ).joinToString(",")
        file.appendText(row + "\n")
    }

    private fun ensureFile(context: Context, name: String, header: List<String>): File {
        val dir = context.getExternalFilesDir(DIRECTORY) ?: context.filesDir
        if (!dir.exists()) dir.mkdirs()
        val file = File(dir, name)
        if (!file.exists()) {
            file.writeText(header.joinToString(",") + "\n")
        }
        return file
    }
}
