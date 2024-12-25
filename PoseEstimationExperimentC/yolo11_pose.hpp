#pragma once
#include "onnx_inference.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>





class yolo11_pose : public onnx::onnx_session {
public:
	yolo11_pose(const wchar_t* model_path, bool use_cuda = false, int device_id = 0) : onnx_session(model_path, use_cuda, device_id) {
		this->set_input_names(&this->input_node_names);
		this->set_output_names(&this->output_node_names);
		this->set_input_dims(this->input_node_dims);
	}

	std::vector<std::vector<keypoint>>  predict(std::vector<cv::Mat>* input);
private:
	std::vector<const char*> input_node_names{ "input" };
	std::vector<const char*> output_node_names{ "outputs" };
	std::vector<int64_t> input_node_dims{ 1, 3, 640,  640 };
	float confidence_threshold = 0.f;

};