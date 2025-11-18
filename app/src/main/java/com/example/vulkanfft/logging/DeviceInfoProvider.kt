package com.example.vulkanfft.logging

import android.content.Context
import android.os.Build
import android.os.PowerManager

data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val hardware: String,
    val board: String,
    val soc: String?,
    val sdkInt: Int,
    val isPowerSaveMode: Boolean
)

object DeviceInfoProvider {
    fun collect(context: Context): DeviceInfo {
        val pm = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        val socInfo = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            Build.SOC_MANUFACTURER?.let { "$it ${Build.SOC_MODEL}" }
        } else {
            null
        }
        return DeviceInfo(
            manufacturer = Build.MANUFACTURER.orEmpty(),
            model = Build.MODEL.orEmpty(),
            hardware = Build.HARDWARE.orEmpty(),
            board = Build.BOARD.orEmpty(),
            soc = socInfo,
            sdkInt = Build.VERSION.SDK_INT,
            isPowerSaveMode = pm.isPowerSaveMode
        )
    }
}
