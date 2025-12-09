package com.example.vulkanfft.util

enum class DelegateType(val displayName: String) {
    GPU("TFLite GPU"),
    NNAPI("TFLite NNAPI"),
    CPU("TFLite CPU");

    override fun toString(): String = displayName
}
