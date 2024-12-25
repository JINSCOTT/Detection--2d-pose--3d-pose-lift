#include "yolo11_pose.hpp"




std::vector<std::vector<keypoint>>  yolo11_pose::predict(std::vector<cv::Mat>* input) {
	// shrink the input
	cv::Mat blob;
	std::vector<cv::Mat>normalized;
	std::vector<Ort::Value> input_tensor, output_tensor;
	std::vector<float >ratios;
	std::vector<cv::Point> original_dim;
	// Preprocess
	cv::Mat padded,current;

	// Adapt to batch size
	this->input_node_dims[0] = input->size();

	for (int i = 0; i < input->size(); i++) {
		original_dim.push_back(cv::Point(input->at(i).cols, input->at(i).rows));
		
		float scale = std::min((float)this->input_node_dims[2] / (float)input->at(i).cols, (float)this->input_node_dims[3] / (float)input->at(i).rows);
		cv::resize(input->at(i), current, cv::Size(), scale, scale);
		cv::copyMakeBorder(current, padded, 0, this->input_node_dims[3] - current.rows, 0, this->input_node_dims[2] - current.cols, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		normalized.push_back(padded);
	
	}
	try {
		blob = cv::dnn::blobFromImages(normalized, 1.0 / 255.0
			, cv::Size(256, 192), cv::Scalar(0, 0, 0), false, false);

	}
	catch (cv::Exception& e) {
		std::cout << e.what();
		system("pause");
	}

	size_t input_tensor_size = blob.total();
	input_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
	// Run model
	try {
		output_tensor = this->run(&input_tensor);
	}
	catch (std::exception& e) {
		throw;
	}
	std::cout << "pose run\n";
	std::cout << "outputsize: " << output_tensor.size() << "\n";
	float* output = output_tensor[0].GetTensorMutableData<float>();
	int64_t* shape = output_tensor[0].GetTensorTypeAndShapeInfo().GetShape().data();

	std::vector<std::vector<keypoint>> detected_keypoints(shape[0], std::vector<keypoint>(shape[1]));

	for (int i = 0; i < detected_keypoints.size(); i++) {
		for (int j = 0; j < detected_keypoints[i].size(); j++) {

			detected_keypoints[i][j].x /= 384.0 * (float)original_dim[i].x;
			detected_keypoints[i][j].y /= 512.0 * (float)original_dim[i].y;
			std::cout << "Keypoint: " << detected_keypoints[i][j].x << ", " << detected_keypoints[i][j].y << ", " << detected_keypoints[i][j].score << "\n";
		}
	}
	return detected_keypoints;
}