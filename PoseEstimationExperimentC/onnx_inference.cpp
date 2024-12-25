# include "onnx_inference.hpp"


namespace onnx{
	void set_cuda(Ort::SessionOptions* options, int cpu_id) {
		if (options == NULL) {
			throw std::invalid_argument("Options pointer is null.\n");
		}
		OrtCUDAProviderOptions cuda_options;

		cuda_options.device_id = cpu_id;
		cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
		cuda_options.arena_extend_strategy = 0;
		cuda_options.do_copy_in_default_stream = 0;
		options->AppendExecutionProvider_CUDA(cuda_options);
	}
	void set_session(Ort::SessionOptions* options, bool use_cuda, int cpu_id) {
		if (options == NULL) {
			throw std::invalid_argument("Options pointer is null.\n");
		}
		options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
		// Set cuda 
		if (use_cuda) {
			try {
				set_cuda(options, cpu_id);
			}
			catch (std::runtime_error e) {
				throw;
			}
			
		}
	}
	std::ostream& operator<<(std::ostream& os, const onnx_session& session)
	{
		os << "Print session\n";
		os << "Input node names: " << session.input_node_names << "\n";
		os << "Output node names: " << session.output_node_names << "\n";
		os << "Input node dimensions: ";
		for (int i = 0; i < session.input_node_dims.size(); i++) {
			os << session.input_node_dims[i] << " ";	
		}
		os << "\n";
		return os;
	}
	onnx_session::onnx_session(const wchar_t* model_path, bool use_cuda, int device_id) {
		// Create environment
		this->env = std::move(Ort::Env(ORT_LOGGING_LEVEL_WARNING));

		Ort::SessionOptions options;
		// Set session options
		try {
			set_session(&options, use_cuda, device_id);
		}
		catch (std::runtime_error e) {
			throw;
		}
		// Try to create session
		try {
			// Model path is const wchar_t*
			session = std::move(Ort::Session(env, model_path, options));
		}
		catch (Ort::Exception oe) {
			throw;
		}
		// Create memory info for input allocations
		try {
			memory_info = std::move(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
		}
		catch (Ort::Exception oe) {
			throw;
		}
	}
	void onnx_session::set_input_names(std::vector<const char*>* input_node_names) {
		this->input_node_names = input_node_names;
	}
	void onnx_session::set_output_names(std::vector<const char*>* output_node_names) {
		this->output_node_names = output_node_names;
	}
	void onnx_session::set_input_dims(const std::vector<int64_t> input_node_dims) {
		this->input_node_dims = input_node_dims;
	}

	std::vector<Ort::Value> onnx_session::run( std::vector<Ort::Value>* input) {
		std::vector<Ort::Value> output;
		 try {
			 output = this->session.Run(Ort::RunOptions{ nullptr }, input_node_names->data(), input->data(), input->size(), output_node_names->data(), output_node_names->size());
		 }
		 catch (Ort::Exception oe) {
			 throw;
		 }
		 return output;
	}
}