package com.example.vulkanfft.view

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.example.vulkanfft.FirstViewModel
import com.example.vulkanfft.databinding.FragmentFirstBinding
import com.example.vulkanfft.util.DelegateType

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

        binding.buttonZiro.setOnClickListener {
            Log.d("MAD" , "IN")
            viewModel.runCPU()
        }

        binding.buttonFirst.setOnClickListener {
            this.context?.let { it1 ->
                viewModel.runBenchmark(
                    it1,
                    delegateType = DelegateType.CPU
                )
            }
        }

        binding.buttonGpu.setOnClickListener {
            this.context?.let { it1 ->
                viewModel.runBenchmark(
                    it1,
                    delegateType = DelegateType.GPU
                )
            }
        }

        binding.buttonNp.setOnClickListener {
            this.context?.let { it1 ->
                viewModel.runBenchmark(
                    it1,
                    delegateType = DelegateType.NNAPI
                )
            }
        }

        binding.buttonFftCpu.setOnClickListener {
            viewModel.runFftCpu()
        }

        binding.buttonFftCpuDelegate.setOnClickListener {
            this.context?.let { context ->
                viewModel.runFftTflite(context, DelegateType.CPU)
            }
        }

        binding.buttonFftGpu.setOnClickListener {
            this.context?.let { context ->
                viewModel.runFftTflite(context, DelegateType.GPU)
            }
        }

        binding.buttonFftNnapi.setOnClickListener {
            this.context?.let { context ->
                viewModel.runFftTflite(context, DelegateType.NNAPI)
            }
        }

        viewModel.result.observe(viewLifecycleOwner) { result ->
            println(result)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
