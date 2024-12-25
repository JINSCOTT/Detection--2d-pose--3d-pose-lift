// yolov7 detection
// reference:
// https://github.com/WongKinYiu/yolov7
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <vector>
#include "onnx_inference.hpp"
#include "utility.hpp"

// detection
class yolov7_det : public onnx::onnx_session {
public:
	yolov7_det(const wchar_t* model_path, bool use_cuda = false, int device_id = 0) : onnx_session(model_path, use_cuda, device_id) {
		this->set_input_names(&this->input_node_names);
		this->set_output_names(&this->output_node_names);
		this->set_input_dims(this->input_node_dims);
	}

	std::vector<bbox> detect( cv::Mat input);
private:
	std::vector<const char*> input_node_names{ "images" };
	std::vector<const char*> output_node_names{ "output" };
	std::vector<int64_t> input_node_dims{ 1, 3, 640, 640 };
	float confidence_threshold = 0.f;
};