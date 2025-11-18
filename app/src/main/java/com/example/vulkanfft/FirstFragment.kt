package com.example.vulkanfft.view

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.core.content.FileProvider
import androidx.core.view.isVisible
import androidx.fragment.app.Fragment
import com.example.vulkanfft.FirstViewModel
import com.example.vulkanfft.databinding.FragmentFirstBinding
import com.example.vulkanfft.logging.ResultLogger
import com.example.vulkanfft.util.DelegateType
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

        binding.buttonMadCpuSingle.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadCpuSingle(ctx) }
        }

        binding.buttonMadTfliteCpuSingle.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadDelegateSingle(ctx, DelegateType.CPU) }
        }

        binding.buttonMadTfliteGpuSingle.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadDelegateSingle(ctx, DelegateType.GPU) }
        }

        binding.buttonMadTfliteNnSingle.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadDelegateSingle(ctx, DelegateType.NNAPI) }
        }

        binding.buttonMadCpuBatch.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadCpuBatch(ctx) }
        }

        binding.buttonMadTfliteCpuBatch.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadBenchmark(ctx, DelegateType.CPU, repetitions = 10) }
        }

        binding.buttonMadTfliteGpuBatch.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadBenchmark(ctx, DelegateType.GPU, repetitions = 10) }
        }

        binding.buttonMadTfliteNnBatch.setOnClickListener {
            this.context?.let { ctx -> viewModel.runMadBenchmark(ctx, DelegateType.NNAPI, repetitions = 10) }
        }

        binding.buttonFftCpu.setOnClickListener {
            this.context?.let { context ->
                viewModel.runFftCpu(context)
            }
        }

        binding.buttonFftTfliteCpu.setOnClickListener {
            this.context?.let { context ->
                viewModel.runFftTflite(context, DelegateType.CPU)
            }
        }

        binding.buttonFftTfliteGpu.setOnClickListener {
            this.context?.let { context ->
                viewModel.runFftTflite(context, DelegateType.GPU)
            }
        }

        binding.buttonFftTfliteNn.setOnClickListener {
            this.context?.let { context ->
                viewModel.runFftTflite(context, DelegateType.NNAPI)
            }
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

        viewModel.result.observe(viewLifecycleOwner) { result ->
            println(result)
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

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
