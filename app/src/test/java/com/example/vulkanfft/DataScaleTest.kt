package com.example.vulkanfft

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DataScaleTest {

    @Test
    fun `default scales stay limited to UI chips`() {
        val expected = listOf(DataScale.BASE, DataScale.DOUBLE, DataScale.QUADRUPLE)
        assertEquals(expected, DataScale.defaultScales)
    }

    @Test
    fun `extreme scales keep ordering and bigger lengths`() {
        val expected = listOf(
            DataScale.EXTREME_32K,
            DataScale.EXTREME_64K,
            DataScale.EXTREME_128K
        )
        assertEquals(expected, DataScale.extremeScales)
        assertTrue(expected.all { it.madVectorLength == it.fftSignalLength })
        val baseMax = DataScale.QUADRUPLE.madVectorLength
        assertTrue(expected.all { it.madVectorLength > baseMax })
    }

    @Test
    fun `focused scales include micro lengths and 128k`() {
        val expected = listOf(
            DataScale.MICRO_512,
            DataScale.MICRO_1K,
            DataScale.MICRO_2K,
            DataScale.EXTREME_128K
        )
        assertEquals(expected, DataScale.focusedScales)
        assertTrue(expected.first().madVectorLength < DataScale.BASE.madVectorLength)
    }

    @Test
    fun `experimental scales list problematic lengths`() {
        val expected = listOf(
            DataScale.EXTREME_256K,
            DataScale.EXTREME_526K
        )
        assertEquals(expected, DataScale.experimentalScales)
        assertTrue(expected.all { it.madVectorLength >= 262_144 })
    }

    @Test
    fun `unified scales cover safe lengths in ascending order`() {
        val unified = DataScale.unifiedScales
        assertEquals(8, unified.size)
        val lengths = unified.map { it.madVectorLength }
        assertTrue(lengths.zipWithNext().all { (a, b) -> a < b })
        assertTrue(lengths.first() == 512)
        assertTrue(lengths.last() == DataScale.EXTREME_64K.madVectorLength)
    }
}
