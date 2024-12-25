#pragma once
// onnx_session class
// Created as base class for onnx model inference
// derive this class to create a session for a specific model
#include "onnxruntime_cxx_api.h"
#include <iostream>
namespace onnx {

	// Set cuda and device id
	void set_cuda(Ort::SessionOptions* options, int device_id);
	// Set session options
	void set_session(Ort::SessionOptions *options,bool use_cuda, int cpu_id);
	/// <summary>
	/// Base clas for onnx session
	/// Derive this class to create a session for a specific modepiranhl
	/// </summary>
	class onnx_session {
	public:
		/// <summary>
		/// Constructor for onnx model inference
		/// </summary>
		/// <param name="model_path">Path to model file</param>
		/// <param name="use_cuda"> whether to use gpu or not</param>
		/// <param name="device_id"> GPU ID, only if you have multiple</param>
		onnx_session(const wchar_t* model_path, bool use_cuda= false, int device_id = 0);
		// Load in input node names
		void set_input_names(std::vector<const char*>* input_node_names);
		// Load in output node names
		void set_output_names(std::vector<const char*>* output_node_names);
		// Load in input node dimensions
		// change dimension 0 to change batch size
		void set_input_dims(const std::vector<int64_t> input_node_dims);
		// Do the actual inference of model
		std::vector<Ort::Value> run(  std::vector<Ort::Value>* input );
		// kind of a debug functionm but not really used
		friend  std::ostream& operator<<(std::ostream& os, const onnx_session& session);
	protected:
		Ort::Env env{ nullptr };
		Ort::Session session{ nullptr };
		Ort::MemoryInfo memory_info{ nullptr };					// Used to allocate memory for input/output nodes
		std::vector<const char*>* output_node_names = nullptr;	// output node names
		std::vector<const char*>* input_node_names = nullptr;	// Input node names
		std::vector<int64_t> input_node_dims;					// input node dimensions
	};
}





