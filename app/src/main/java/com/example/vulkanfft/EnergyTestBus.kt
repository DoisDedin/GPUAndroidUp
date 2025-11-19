package com.example.vulkanfft

import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow

object EnergyTestBus {

    private val _events = MutableSharedFlow<EnergyTestEvent>(
        replay = 0,
        extraBufferCapacity = 8,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val events: SharedFlow<EnergyTestEvent> = _events

    suspend fun emit(event: EnergyTestEvent) {
        _events.emit(event)
    }

    fun tryEmit(event: EnergyTestEvent) {
        _events.tryEmit(event)
    }
}

sealed class EnergyTestEvent {
    object Started : EnergyTestEvent()
    data class Status(
        val current: Int,
        val total: Int,
        val scenarioLabel: String,
        val percentComplete: Int,
        val scale: DataScale
    ) : EnergyTestEvent()
    data class Progress(val snapshot: BenchmarkExecutor.EnergyScenarioSnapshot) : EnergyTestEvent()
    data class Finished(val summary: String) : EnergyTestEvent()
    data class Error(val message: String) : EnergyTestEvent()
    object Cancelled : EnergyTestEvent()
}
