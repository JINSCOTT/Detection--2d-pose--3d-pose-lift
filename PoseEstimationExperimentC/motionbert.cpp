#include "motionbert.hpp"


std::vector<keypoint> Halpe26_to_h36(std::vector<keypoint> keypoints) {
	std::vector<keypoint> coco_keypoints;
	// hip
	coco_keypoints.push_back((keypoints[11] + keypoints[12]) / 2);
	// rhip
	coco_keypoints.push_back(keypoints[12]);
	// rknee
	coco_keypoints.push_back(keypoints[14]);
	// rfoot
	coco_keypoints.push_back(keypoints[16]);
	// lhip
	coco_keypoints.push_back(keypoints[11]);
	// lknee
	coco_keypoints.push_back(keypoints[13]);
	// lfoot
	coco_keypoints.push_back(keypoints[15]);
	// belly
	coco_keypoints.push_back((keypoints[5] + keypoints[6] + keypoints[11] + keypoints[12]) / 4);
	// neck
	coco_keypoints.push_back(keypoints[18]);
	// nose
	coco_keypoints.push_back(keypoints[0]);
	// head
	coco_keypoints.push_back(keypoints[17]);
	// lshoulder
	coco_keypoints.push_back(keypoints[5]);
	// lelbow
	coco_keypoints.push_back(keypoints[7]);
	// lhand
	coco_keypoints.push_back(keypoints[9]);
	// rshoulder
	coco_keypoints.push_back(keypoints[6]);
	// relbow
	coco_keypoints.push_back(keypoints[8]);
	// rhand
	coco_keypoints.push_back(keypoints[10]);
	return coco_keypoints;
}

std::vector<std::vector<cv::Point3f>>  motionbert::predict(std::vector<std::vector<keypoint>> keypoints, float original_w, float original_h) {
	// Preprocess
	std::vector<Ort::Value> input_tensor, output_tensor;
	std::vector<std::vector<cv::Point3f>>keypoint_3d;
	float scale = std::min(original_w, original_h) / 2.0f;

	// Assemble the input to insure data is contiguous
	float input_array[10][17][3] = { 0 };

	// Preprocess
	for (int i = 0; i < keypoints.size(); i++) {
		for (int j = 0; j < keypoints[i].size(); j++) {
			input_array[i][j][0] = (keypoints[i][j].x - original_w / 2.0) / scale;
			input_array[i][j][1] = (keypoints[i][j].y - original_h / 2.0) / scale;
		}
	}
	size_t input_tensor_size = input_node_dims[0] * input_node_dims[1] * input_node_dims[2] * input_node_dims[3];
	input_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)input_array, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
	// Run model
	try {
		output_tensor = this->run(&input_tensor);
	}
	catch (std::exception& e) {
		throw;
	}
	const float* result = output_tensor.front().GetTensorData<float>();

	// Assemble the result
	for (int i = 0; i < this->input_node_dims[1]; i++) {
		std::vector<cv::Point3f> keypoint_3d_frame;
		for (int j = 0; j < this->input_node_dims[2]; j++) {
			cv::Point3f keypoint;
			int index = i * this->input_node_dims[2] * this->input_node_dims[3] + j * this->input_node_dims[3];
			keypoint.x = result[index];
			keypoint.y = result[index + 1];
			keypoint.z = result[index + 2];
			keypoint_3d_frame.push_back(keypoint);
		}
		keypoint_3d.push_back(keypoint_3d_frame);

	}
	return keypoint_3d;
}