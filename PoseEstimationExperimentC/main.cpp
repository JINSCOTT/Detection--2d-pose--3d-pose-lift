/*
C program for 3d pose estimation
flow: detect-> 2d pose estimation -> 3d pose lift


*/
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <deque>
#include "yolov7.hpp"
#include "rtmpose.hpp"
#include "motionbert.hpp"
void main() {

	//Models
	std::unique_ptr<yolov7_det> yolov7_det_model;
	std::unique_ptr<rtmpose> rtmpose_model;
	std::unique_ptr<motionbert> motionbert_model;

	cv::VideoCapture cap;
	// Main frame
	cv::Mat Frame;

	// 2d keypoints of "multiple persons" 
	std::vector<std::vector<keypoint>> keypoints;
	// buffer of 2d keypoint of a "single object"
	std::vector<std::vector<keypoint>> keypoint_buffer;
	// Result of motion bert 10 instances of keypoints, only use the first one
	std::vector<std::vector<cv::Point3f>> keypoints_3d;


	// Load models
	// Detection
	// change the true to false if you want to use cpu, rather than gpu
	try {
		yolov7_det_model = std::make_unique<yolov7_det>(L"yolov7-tiny.onnx", true, 0);
	}
	catch (std::exception& e) {
		std::cout << e.what();
		return;
	}
	// 2d pose estimation
	try {
		rtmpose_model = std::make_unique<rtmpose>(L"rtmpose.onnx", true, 0);
	}
	catch (std::exception& e) {
		std::cout << e.what();
		return;
	}
	// 2d to 3d pose lifting
	try {
		motionbert_model = std::make_unique < motionbert>(L"motion_bert_opt.onnx", true, 0);
	}
	catch (std::exception& e) {
		std::cout << e.what();
		return;
	}

	// Open cap
	bool error = cap.open(0, cv::CAP_ANY);
	// Main loop
	while (error) {
		// Read frame
		error = cap.read(Frame);
		// create hard frame copy
		// use Frame to draw result and get clean image from frame_copy
		cv::Mat frame_copy = Frame.clone();

		// Run the model
		std::vector<bbox> objs = yolov7_det_model->detect(frame_copy.clone());
		// Draw the detected objects
		for (int i = 0; i < objs.size(); i++) {
			cv::rectangle(Frame, cv::Rect(objs.at(i).x, objs.at(i).y, objs.at(i).w, objs.at(i).h), cv::Scalar(0, 255, 255), 2);
			cv::putText(Frame, coco_label_tostring(objs.at(i).class_id), cv::Point(objs.at(i).x, objs.at(i).y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
		}
		// Now we want to focus on person only
		std::vector<bbox> person_objs;
		for (int i = 0; i < objs.size(); i++) {
			if (objs.at(i).class_id == coco_label::PERSON) {
				person_objs.push_back(objs.at(i));
			}
		}
		// 2d position estimation
		if (person_objs.size() > 0) {
			keypoints = rtmpose_model->predict(frame_copy.clone(), person_objs);
			// Draw out all 2d keypoints
			for (int i = 0; i < keypoints.size(); i++) {
				for (int j = 0; j < keypoints[i].size(); j++) {
					if (keypoints[i][j].score > 0.4f) {
						cv::circle(Frame, cv::Point(keypoints[i][j].x, keypoints[i][j].y), 2, cv::Scalar(0, 0, 255), 2);
						cv::putText(Frame, std::to_string(j), cv::Point(keypoints[i][j].x, keypoints[i][j].y), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
					}

				}
			}
			// I didn't implement any tracking, so i assume 0 is always the same one
			std::vector<keypoint> keypoints_coco = Halpe26_to_h36(keypoints[0]);
			keypoint_buffer.push_back(keypoints_coco);
		}

		// 3d pose estimation
		float  scale = std::min(Frame.cols, Frame.rows) / 2.0f;
		// This is because the model is 1x10x17x3
		if (keypoint_buffer.size() == 10) {
			// Predict
			keypoints_3d = motionbert_model->predict(keypoint_buffer, Frame.cols, Frame.rows);
			for (int i = 0; i < keypoints_3d[0].size(); i++) {
				// Reproject 3d pos bect two 2d, this is to show the the projection is correct
				cv::Point pos = cv::Point(keypoints_3d[0][i].x * scale + Frame.cols / 2.0f, keypoints_3d[0][i].y * scale + Frame.rows / 2.0f);
				cv::putText(Frame, std::to_string(i), pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 2);
				cv::circle(Frame, pos, 2, cv::Scalar(255, 0, 255), 2);
			}
			keypoint_buffer.erase(keypoint_buffer.begin());
		}

		// Display the result
		cv::imshow("Frame", Frame);
		//cv::waitKey(0);
		if (cv::waitKey(1) == 27) break;
	}
	cap.release();
}