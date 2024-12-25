// motion bert model 
// reference: https://github.com/Walter0807/MotionBERT
// generated with torch dynamo
// lifts human36m keypoints to 3d
// input: 10x17x3
// output: 10x17x3
// Although the model returns 10x17x3, only the fist iteration is used
// And with this we can expect about 330 to 400 ms latency, which is quite large
// generally these these types of models prioritize accuracy over latency

#pragma once
#include "onnx_inference.hpp"
#include "utility.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>

// convert halpe26 to human36m joint points
// Some points joint does not have direct mapping so averege are used by soim,e
std::vector<keypoint> Halpe26_to_h36(std::vector<keypoint> keypoints);

class motionbert : public onnx::onnx_session {
public:
	// Constructor for motion bert model
	motionbert(const wchar_t* model_path, bool use_cuda = false, int device_id = 0) : onnx_session(model_path, use_cuda, device_id) {
		this->set_input_names(&this->input_node_names);
		this->set_output_names(&this->output_node_names);
		this->set_input_dims(this->input_node_dims);
	}
	/// <summary>
	/// Predict 3d keypoints from 2d keypoints
	/// </summary>
	/// <param name="">buffer 2d keypoint for 1 times</param>
	/// <param name="original_w">width</param>
	/// <param name="original_h">height</param>
	/// <returns></returns>
	std::vector<std::vector<cv::Point3f>>  predict(std::vector<std::vector<keypoint>> keypoint_buffer, float original_w, float original_h);
	
private:
	std::vector<const char*> input_node_names{ "input" };
	std::vector<const char*> output_node_names{ "view_289" };
	std::vector<int64_t> input_node_dims{ 1, 10,17,3 };
};