#pragma once
// rtmpose model
// https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose
// https://github.com/Tau-J/rtmlib
// I didn't want to use their c implementation in the beginning, but I can't fugure out the correct preprocessing code, so in the end, the part calculating 
// mean and std has to be used from their c code, otherwise most are converted from python 
#include "onnx_inference.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include "utility.hpp"

// Rotate point around origin
cv::Point rotate_point(cv::Point p, float angle);

/*
The 3rd point is defined by rotating vector `a - b` by 90 degrees
anticlockwise, using b as the rotation center.
*/
cv::Point get_3rd_point(cv::Point a, cv::Point b);

// Transform the bbox format from (x,y,w,h) into (center, scale)
void bbox_xyxy2cs(std::vector<bbox>* objs, float padding , std::vector<cv::Point> &center, std::vector<cv::Point>&scale );

// Get the bbox image as the model input by affine transform.
void top_down_affine(int64_t* dims, std::vector<cv::Point>& scale, std::vector<cv::Point>& center,cv::Mat Frame, std::vector<cv::Mat>&resized_image);

//Calculate the affine transformation matrix that can warp the bbox area
// in the input image to the output size.
void get_warp_matrix(std::vector<cv::Point>& center, std::vector<cv::Point>& scale, float rot, float w, float h, std::vector<cv::Mat>& affine_matrix, bool inverse = false);

// "Get maximum response location and value from simcc representations.
std::vector<std::vector<keypoint>> get_simcc_maximum(const float* simcc_x, const  float* simcc_y, int64_t* dimx, int64_t* dimy);

// rtmpose model
class rtmpose : public onnx::onnx_session {
public:
	rtmpose(const wchar_t* model_path, bool use_cuda = false, int device_id = 0) : onnx_session(model_path, use_cuda, device_id) {
		this->set_input_names(&this->input_node_names);
		this->set_output_names(&this->output_node_names);
		this->set_input_dims(this->input_node_dims);
	}

	std::vector<std::vector<keypoint>>  predict(cv::Mat frame, std::vector<bbox> box);
private:
	std::vector<const char*> input_node_names{ "input" };
	std::vector<const char*> output_node_names{ "simcc_x", "simcc_y"};
	std::vector<int64_t> input_node_dims{ 1, 3, 256,  192 };
	cv::Scalar mean{ 123.675, 116.28, 103.53 };
	cv::Scalar std{ 58.395, 57.12, 57.375 };
};