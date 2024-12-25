#include "rtmpose.hpp"


cv::Point rotate_point(cv::Point p, float angle) {
	float sn = sin(angle);
	float cs = cos(angle);

	// rotate point
	float xnew = p.x * cs - p.y * sn;
	float ynew = p.x * sn + p.y * cs;
	// translate point back:

	return cv::Point(xnew, ynew);
}

cv::Point get_3rd_point(cv::Point a, cv::Point b) {
	cv::Point dir = a - b;
	return b + cv::Point(-dir.y, dir.x);
}


void bbox_xyxy2cs(std::vector<bbox>* objs, float padding , std::vector<cv::Point>& center, std::vector<cv::Point>& scale) {
	for (int i = 0; i < objs->size(); i++) {
		// Calculate center
		center.push_back(cv::Point(objs->at(i).x + objs->at(i).w / 2.0, objs->at(i).y + objs->at(i).h / 2.0));
		// Calculate scale
		scale.push_back(cv::Point(objs->at(i).w * padding, objs->at(i).h * padding));
	}
}

void top_down_affine(int64_t* dims, std::vector<cv::Point>& scale, std::vector<cv::Point>& center, cv::Mat Frame, std::vector<cv::Mat>& resized_image) {

	float w = dims[3], h = dims[2];
	float aspect_ratio = w / h;
	// reshape bbox to fixed aspect ratio
	for (int i = 0; i < scale.size(); i++) {
		if (scale[i].x > scale[i].y * aspect_ratio) {
			scale[i].y = scale[i].x / aspect_ratio;
		}
		else {
			scale[i].x = scale[i].y * aspect_ratio;
		}
	}
	// get the affine matrix
	float rot = 0;
	std::vector<cv::Mat> affine_matrix;
	get_warp_matrix(center, scale, rot, w,h , affine_matrix);

	// warp and push image
	for (int i = 0; i < affine_matrix.size(); i++) {
		cv::Mat resized;
		cv::warpAffine(Frame.clone(), resized, affine_matrix[i], cv::Size(w, h), cv::INTER_LINEAR);
		resized_image.push_back(resized);
	}
}

// Create transformation matrix
void get_warp_matrix(std::vector<cv::Point>& center, std::vector<cv::Point>& scale, float rot, float w, float h, std::vector<cv::Mat>& affine_matrix, bool inverse) {

	cv::Point shift = cv::Point(0.,0.);
	for (int i = 0; i < center.size(); i++) {
		float src_w = scale[i].x;
		float dst_w = w;
		float dst_h = h;

		// compute transormation matrix
		float rot_rad = rot * 3.14159265 / 180.0;
		cv::Point src_dir = rotate_point(cv::Point(0., src_w*-0.5), rot_rad);
		cv::Point dst_dir = cv::Point(0., dst_w * -0.5);

		// get four corners of the src rectangle in the original image
		cv::Point2f src[3];
		src[0] = cv::Point(center[i].x + scale[i].x * shift.x, center[i].y + scale[i].y * shift.y);
		src[1] = cv::Point(center[i].x + src_dir.x + scale[i].x * shift.x, center[i].y + src_dir.y + scale[i].y * shift.y);
		src[2] = get_3rd_point(src[0], src[1]);

		//get four corners of the dst rectangle in the input image
		cv::Point2f dst[3];
		dst[0] = cv::Point(dst_w * 0.5, dst_h * 0.5);
		dst[1] = cv::Point(dst_w * 0.5 + dst_dir.x, dst_h * 0.5 + dst_dir.y);
		dst[2] = get_3rd_point(dst[0], dst[1]);

		cv::Mat result;
		try {
			if (!inverse) {
				result = cv::getAffineTransform(src, dst);
			}
			else {
				result = cv::getAffineTransform(dst, src);
			}
		}
		catch (cv::Exception& e) {
			throw;
		}
		
		// Push back result
		affine_matrix.push_back(result);
	}


}

std::vector<std::vector<keypoint>> get_simcc_maximum(const float* simcc_x, const  float* simcc_y, int64_t* dimx, int64_t* dimy) {

	// Find the highest x and y

	int N = dimx[0], K = dimx[1], Wx = dimx[2], Wy = dimy[2];
	std::vector<std::vector<keypoint>> keypoints(N, std::vector<keypoint>(K));

	int i = 0, j = 0, k = 0, loc = 0;
	float max = -1;
	for (i = 0; i < N; i++) {
		for (j = 0; j < K; j++) {
			max = -1;
			for (k = 0; k < Wx; k++) {
				if (simcc_x[i * K * Wx + j * Wx + k] > max) {
					max = simcc_x[i * K * Wx + j * Wx + k];
					loc = k;
				}
			}
			keypoints[i][j].x = loc;
			keypoints[i][j].score = max * 0.5f;
			max = -1;
			for (k = 0; k < Wy; k++) {
				if (simcc_y[i * K * Wy + j * Wy + k] > max) {
					max = simcc_y[i * K * Wy + j * Wy + k];
					loc = k;
				}
			}
			keypoints[i][j].y = loc;
			keypoints[i][j].score += max * 0.5f;
		}
	}
	return keypoints;
}

std::vector<std::vector<keypoint>>  rtmpose::predict(cv::Mat frame, std::vector<bbox> box) {
	std::vector<std::vector<keypoint>> result;

	// There is nothing
	if (box.size() == 0) {
		return result;
	}
		
	// Calculate center and scale
	std::vector<cv::Point> center, scale;
	std::vector<cv::Mat> resized_image;
	std::vector<cv::Mat> affine_transform_reverse;
	cv::Mat blob;
	std::vector<Ort::Value> input_tensor, output_tensor;
	
	// Get inputs
	bbox_xyxy2cs(&box, 1.25f, center, scale);
	// get the cropped and warped images
	top_down_affine(this->input_node_dims.data(), scale, center, frame, resized_image);

	// This is used to reverse the affine transformation to original space 
	get_warp_matrix(center, scale, 0, this->input_node_dims[3], this->input_node_dims[2], affine_transform_reverse, true);


	// Adapt to batch size
	this->input_node_dims[0] = resized_image.size();

	// Preprocess
	for (int i = 0; i < resized_image.size(); i++) {
		cv::cvtColor(resized_image.at(i), resized_image.at(i), cv::COLOR_BGR2RGB);
	}
	std::vector<float> data_array;
	size_t input_tensor_size = input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3];
	data_array.resize(input_tensor_size);

	// Ref https://github.com/HW140701/RTMPose-Deploy/blob/main/Windows/OnnxRumtime-CPU/src/RTMPoseOnnxRuntime/rtmpose_onnxruntime.cpp
	for (int n = 0; n < input_node_dims[0]; n++)
	{
		for (int h = 0; h < input_node_dims[2]; h++){
			for (int w = 0; w < input_node_dims[3]; w++) {
				for (int c = 0; c < input_node_dims[1]; c++) {
					int chw_index = n * input_node_dims[1] * input_node_dims[2] * input_node_dims[3] + c * input_node_dims[2] * input_node_dims[3] + h * input_node_dims[3] + w;
					float tmp = resized_image[n].ptr<uchar>(h)[w * 3 + c];
					data_array[chw_index] = (tmp - this->mean[c]) / this->std[c];
				}
			}
		}
	}

	// Assemble input tensor
	input_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, data_array.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size()));
	// Run model
	try {
		output_tensor = this->run(&input_tensor);
	}
	catch (std::exception& e) {
		throw;
	}
	std::vector<std::vector<keypoint>> detected_keypoints = get_simcc_maximum(output_tensor[0].GetTensorData<float>(), output_tensor[1].GetTensorData<float>(), output_tensor[0].GetTensorTypeAndShapeInfo().GetShape().data(), output_tensor[1].GetTensorTypeAndShapeInfo().GetShape().data());
	
	// Assemble the result
	for (int i = 0; i < detected_keypoints.size(); i++) {
		for (int j = 0; j < detected_keypoints[i].size(); j++) {
			cv::Mat origin_point_Mat = cv::Mat::ones(3, 1, CV_64FC1);
			origin_point_Mat.at<double>(0, 0) = detected_keypoints[i][j].x/2;
			origin_point_Mat.at<double>(1, 0) = detected_keypoints[i][j].y/2;
			cv::Mat temp_result_mat = affine_transform_reverse[i] * origin_point_Mat;
			detected_keypoints[i][j].x = temp_result_mat.at<double>(0, 0);
			detected_keypoints[i][j].y = temp_result_mat.at<double>(1, 0);		
		}
	}
	return detected_keypoints;
}