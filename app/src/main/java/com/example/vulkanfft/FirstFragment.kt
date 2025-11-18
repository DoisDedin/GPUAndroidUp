package com.example.vulkanfft.view

import android.content.Intent
import android.os.Bundle
import android.os.PowerManager
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import com.example.vulkanfft.BenchmarkScenario
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

        binding.buttonRunEnergyTest.setOnClickListener {
            val context = context ?: return@setOnClickListener
            viewModel.startEnergyTest(context)
        }

        binding.buttonCancelEnergyTest.setOnClickListener {
            viewModel.cancelEnergyTest()
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
                viewModel.runScenario(ctx, scenario)
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
}
