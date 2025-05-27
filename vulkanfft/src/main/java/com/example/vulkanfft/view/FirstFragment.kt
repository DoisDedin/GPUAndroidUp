package com.example.vulkanfft.view

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.navigation.fragment.findNavController
import com.example.vulkanfft.util.SumProcessorGPU
import com.example.vulkanfft.viewmodels.FirstViewModel
import com.seuprojeto.vkfftlib.databinding.FragmentFirstBinding

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

        binding.buttonFirst.setOnClickListener {
            this.context?.let { it1 ->
                viewModel.runBenchmark(
                    it1,
                    delegateType = SumProcessorGPU.DelegateType.CPU
                )
            }
        }

        binding.buttonGpu.setOnClickListener {
            this.context?.let { it1 ->
                viewModel.runBenchmark(
                    it1,
                    delegateType = SumProcessorGPU.DelegateType.GPU
                )
            }
        }

        binding.buttonNp.setOnClickListener {
            this.context?.let { it1 ->
                viewModel.runBenchmark(
                    it1,
                    delegateType = SumProcessorGPU.DelegateType.NNAPI
                )
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