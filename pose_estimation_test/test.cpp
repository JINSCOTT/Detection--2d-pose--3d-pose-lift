#include "pch.h"
#include "../PoseEstimationExperimentC/onnx_inference.hpp"	
TEST(TestCaseName, TestName) {
  EXPECT_EQ(1, 1);
  EXPECT_TRUE(true);
}

// ONNX env should always be called first
 TEST(ONNX, ENV_Startup) {
	//onnx::onnx_env env;
	EXPECT_TRUE(true);
}