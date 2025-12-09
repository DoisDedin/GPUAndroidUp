package com.example.vulkanfft.view

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.PowerManager
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import com.example.vulkanfft.Algorithm
import com.example.vulkanfft.BenchmarkScenario
import com.example.vulkanfft.DataScale
import com.example.vulkanfft.FirstViewModel
import com.example.vulkanfft.R
import com.example.vulkanfft.databinding.FragmentFirstBinding
import com.example.vulkanfft.logging.ResultLogger
import java.util.ArrayList

/**
 * A simple [Fragment] subclass as the default destination in the navigation.
 */
class FirstFragment : Fragment() {

    private lateinit var viewModel: FirstViewModel

    private var _binding: FragmentFirstBinding? = null
    private var pendingEnergyStart = false
    private var pendingMadScale: DataScale = DataScale.BASE
    private var pendingFftScale: DataScale = DataScale.BASE

    private val notificationPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            val ctx = context
            if (granted) {
                if (pendingEnergyStart && ctx != null) {
                    viewModel.startEnergyTest(ctx, pendingMadScale, pendingFftScale)
                }
            } else if (pendingEnergyStart) {
                Toast.makeText(
                    ctx,
                    getString(R.string.energy_status_permission_denied),
                    Toast.LENGTH_LONG
                ).show()
            }
            pendingEnergyStart = false
        }

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {

        _binding = FragmentFirstBinding.inflate(inflater, container, false)
        return binding.root

    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        viewModel = androidx.lifecycle.ViewModelProvider(this)[FirstViewModel::class.java]
        setupIterationControls()
        setupBatchControls()

        binding.buttonRunAll.setOnClickListener {
            this.context?.let { ctx -> viewModel.runFullBenchmarkSuite(ctx) }
        }

        binding.buttonMadCpuSingle.bindScenario(BenchmarkScenario.MAD_CPU_SINGLE)
        binding.buttonMadTfliteCpuSingle.bindScenario(BenchmarkScenario.MAD_TFLITE_CPU_SINGLE)
        binding.buttonMadTfliteGpuSingle.bindScenario(BenchmarkScenario.MAD_TFLITE_GPU_SINGLE)
        binding.buttonMadTfliteNnSingle.bindScenario(BenchmarkScenario.MAD_TFLITE_NN_SINGLE)
        binding.buttonMadCpuBatch.bindScenario(BenchmarkScenario.MAD_CPU_BATCH)
        binding.buttonMadTfliteCpuBatch.bindScenario(BenchmarkScenario.MAD_TFLITE_CPU_BATCH)
        binding.buttonMadTfliteGpuBatch.bindScenario(BenchmarkScenario.MAD_TFLITE_GPU_BATCH)
        binding.buttonMadTfliteNnBatch.bindScenario(BenchmarkScenario.MAD_TFLITE_NN_BATCH)
        binding.buttonFftCpu.bindScenario(BenchmarkScenario.FFT_CPU_SINGLE)
        binding.buttonFftTfliteCpu.bindScenario(BenchmarkScenario.FFT_TFLITE_CPU_SINGLE)
        binding.buttonFftTfliteGpu.bindScenario(BenchmarkScenario.FFT_TFLITE_GPU_SINGLE)
        binding.buttonFftTfliteNn.bindScenario(BenchmarkScenario.FFT_TFLITE_NN_SINGLE)
        binding.buttonFftCpuBatch.bindScenario(BenchmarkScenario.FFT_CPU_BATCH)
        binding.buttonFftTfliteCpuBatch.bindScenario(BenchmarkScenario.FFT_TFLITE_CPU_BATCH)
        binding.buttonFftTfliteGpuBatch.bindScenario(BenchmarkScenario.FFT_TFLITE_GPU_BATCH)
        binding.buttonFftTfliteNnBatch.bindScenario(BenchmarkScenario.FFT_TFLITE_NN_BATCH)

        binding.buttonRunEnergyTest.setOnClickListener { startEnergyTestWithPermission() }

        binding.buttonCancelEnergyTest.setOnClickListener {
            val ctx = context ?: return@setOnClickListener
            viewModel.cancelEnergyTest(ctx)
        }

        binding.buttonShareLogs.setOnClickListener {
            this.context?.let { context ->
                val dir = ResultLogger.logsDir(context)
                val files = dir.listFiles()?.filter { it.isFile } ?: emptyList()
                if (files.isEmpty()) {
                    Toast.makeText(context, "Nenhum log para compartilhar.", Toast.LENGTH_SHORT).show()
                } else {
                    val uris = files.map {
                        FileProvider.getUriForFile(
                            context,
                            "${context.packageName}.fileprovider",
                            it
                        )
                    }
                    val shareIntent = Intent(Intent.ACTION_SEND_MULTIPLE).apply {
                        type = "*/*"
                        putParcelableArrayListExtra(Intent.EXTRA_STREAM, ArrayList(uris))
                        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    startActivity(Intent.createChooser(shareIntent, "Compartilhar resultados"))
                }
            }
        }

        binding.buttonClearLogs.setOnClickListener {
            this.context?.let { context ->
                val success = ResultLogger.clearAll(context)
                val message = if (success) "Logs apagados." else "Não foi possível apagar todos os logs."
                Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
                if (success) {
                    viewModel.resetSessionCounter()
                }
            }
        }

        viewModel.benchmarkResult.observe(viewLifecycleOwner) { text ->
            binding.textBenchmarkResult.text = text
        }

        viewModel.energyResult.observe(viewLifecycleOwner) { text ->
            binding.textEnergyResult.text = text
        }

        viewModel.energyInProgress.observe(viewLifecycleOwner) { running ->
            binding.buttonRunEnergyTest.isEnabled = running != true
            binding.buttonCancelEnergyTest.isEnabled = running == true
            binding.buttonCancelEnergyTest.alpha = if (running == true) 1f else 0.7f
            updateEnergyStatus(running == true)
        }

        viewModel.sessionRunCount.observe(viewLifecycleOwner) { count ->
            binding.textSessionRuns.text = getString(R.string.session_runs_value, count)
        }

        viewModel.progress.observe(viewLifecycleOwner) { state ->
            if (state == null) {
                binding.progressIndicator.isVisible = false
                binding.progressLabel.isVisible = false
                return@observe
            }
            binding.progressIndicator.max = if (state.total <= 0) 1 else state.total
            val current = state.current.coerceAtMost(state.total)
            binding.progressIndicator.setProgressCompat(current, true)
            binding.progressIndicator.isVisible = state.running
            binding.progressLabel.isVisible = state.running
            binding.progressLabel.text =
                "Etapa $current/${state.total} - ${state.message}"
            if (!state.running) {
                binding.progressIndicator.isVisible = false
                binding.progressLabel.isVisible = false
            }
        }
    }

    override fun onResume() {
        super.onResume()
        updateEnergyStatus(viewModel.energyInProgress.value == true)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }

    private fun View.bindScenario(scenario: BenchmarkScenario) {
        setOnClickListener {
            this@FirstFragment.context?.let { ctx ->
                val scale = when (scenario.algorithm) {
                    Algorithm.MAD -> currentMadScale()
                    Algorithm.FFT -> currentFftScale()
                }
                viewModel.runScenario(ctx, scenario, scale = scale)
            }
        }
    }

    private fun updateEnergyStatus(running: Boolean) {
        val ctx = context ?: return
        val pm = ctx.getSystemService(PowerManager::class.java)
        val powerSaveText = if (pm.isPowerSaveMode) {
            getString(R.string.power_save_on)
        } else {
            getString(R.string.power_save_off)
        }
        binding.textPowerSaveHint.text = getString(R.string.energy_note)
        val statusRes = if (running) {
            R.string.energy_status_running
        } else {
            R.string.energy_status_idle
        }
        binding.textEnergyStatus.text =
            getString(statusRes) + "\n" + getString(R.string.energy_mode_state, powerSaveText)
    }

    private fun startEnergyTestWithPermission() {
        val ctx = context ?: return
        val madScale = currentMadScale()
        val fftScale = currentFftScale()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val hasPermission = ContextCompat.checkSelfPermission(
                ctx,
                Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            if (!hasPermission) {
                pendingEnergyStart = true
                pendingMadScale = madScale
                pendingFftScale = fftScale
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                return
            }
        }
        viewModel.startEnergyTest(ctx, madScale, fftScale)
    }

    private fun setupIterationControls() {
        binding.chipGroupIterations.check(iterationChipIdFor(viewModel.currentBenchmarkIterations()))
        binding.chipGroupIterations.setOnCheckedStateChangeListener { _, checkedIds ->
            val id = checkedIds.firstOrNull() ?: return@setOnCheckedStateChangeListener
            iterationValueForChip(id)?.let { viewModel.updateBenchmarkIterations(it) }
        }
    }

    private fun setupBatchControls() {
        binding.chipGroupBatch.check(batchChipIdFor(viewModel.currentBatchSize()))
        binding.chipGroupBatch.setOnCheckedStateChangeListener { _, checkedIds ->
            val id = checkedIds.firstOrNull() ?: return@setOnCheckedStateChangeListener
            batchValueForChip(id)?.let { viewModel.updateBatchSize(it) }
        }
    }

    private fun iterationValueForChip(id: Int): Int? = when (id) {
        binding.chipIterations1.id -> 1
        binding.chipIterations4.id -> 4
        binding.chipIterations8.id -> 8
        binding.chipIterations12.id -> 12
        else -> null
    }

    private fun iterationChipIdFor(value: Int): Int = when (value) {
        1 -> binding.chipIterations1.id
        4 -> binding.chipIterations4.id
        8 -> binding.chipIterations8.id
        12 -> binding.chipIterations12.id
        else -> binding.chipIterations4.id
    }

    private fun batchValueForChip(id: Int): Int? = when (id) {
        binding.chipBatch1.id -> 1
        binding.chipBatch4.id -> 4
        binding.chipBatch8.id -> 8
        binding.chipBatch12.id -> 12
        else -> null
    }

    private fun batchChipIdFor(value: Int): Int = when (value) {
        1 -> binding.chipBatch1.id
        4 -> binding.chipBatch4.id
        8 -> binding.chipBatch8.id
        12 -> binding.chipBatch12.id
        else -> binding.chipBatch4.id
    }

    private fun currentMadScale(): DataScale {
        return when (binding.chipGroupMadScale.checkedChipId) {
            binding.chipMadScaleDouble.id -> DataScale.DOUBLE
            binding.chipMadScaleQuad.id -> DataScale.QUADRUPLE
            else -> DataScale.BASE
        }
    }

    private fun currentFftScale(): DataScale {
        return when (binding.chipGroupFftScale.checkedChipId) {
            binding.chipFftScaleDouble.id -> DataScale.DOUBLE
            binding.chipFftScaleQuad.id -> DataScale.QUADRUPLE
            else -> DataScale.BASE
        }
    }
}
