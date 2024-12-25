#include"yolov7.hpp"

std::string coco_label_tostring(coco_label label) {
	return labelname.at(label); // Throws std::out_of_range if color is not found
}

std::vector<bbox> yolov7_det::detect(cv::Mat input) {

	std::vector<Ort::Value> input_tensor, output_tensor;
	std::vector<bbox> detected_bboxes;
	// Preprocess
	cv::Mat blob = cv::dnn::blobFromImage(input.clone(), 1 / 255.0, cv::Size(640, 640), (0, 0, 0), false, false);
	size_t input_tensor_size = blob.total();
	input_tensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)blob.data, input_tensor_size, input_node_dims.data(), input_node_dims.size()));
	
	// Run model
	try {
		output_tensor = this->run(&input_tensor);
	}
	catch (std::exception& e) {
		throw;
	}
	const float* result = output_tensor.front().GetTensorData<float>();
	int result_size = output_tensor.front().GetTensorTypeAndShapeInfo().GetElementCount();

	// Calculate results to original space
	for (int i = 0; i < result_size; i += 7) {		// the output of this model is (number_of_detected , 7), thus the increment by 7
		float x = std::max(result[i + 1] / 640.0 * (float)input.cols, 0.0);
		float y = std::max(result[i + 2] / 640.0 * (float)input.rows, 0.0);
		float w = (result[i + 3] - result[i + 1]) / 640.0 * (float)input.cols;
		float h = (result[i + 4] - result[i + 2]) / 640.0 * (float)input.rows;
		if (x + w >= (float)input.cols)w = (float)input.cols - x;
		if (y + h >= (float)input.rows)h = (float)input.rows - y;
		int class_id = result[i + 5];
		float confidence = result[i + 6];
		
		// Push_back
		detected_bboxes.push_back(bbox(x, y, w, h, input.cols, input.rows, confidence, static_cast<coco_label>(class_id)));
	}
	return detected_bboxes;
}

