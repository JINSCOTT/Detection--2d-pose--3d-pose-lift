# pragma once
// utility to pass data
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <iostream>
#include <vector>

// 2
class keypoint {
public:
	float x;
	float y;
	float score;
	keypoint operator+(keypoint const& obj)
	{
		keypoint res;
		res.x = this->x + obj.x;
		res.y = this->y + obj.y;
		res.score = this->score + obj.score;
		return res;
	}
	keypoint operator/(float const& obj)
	{
		keypoint res;
		res.x = this->x/obj;
		res.y = this->y / obj;
		res.score = this->score / obj;
		return res;
	}

};

enum class coco_label {
	PERSON = 0,
	BICYCLE = 1,
	CAR = 2,
	MOTORBIKE = 3,
	AEROPLANE = 4,
	BUS = 5,
	TRAIN = 6,
	TRUCK = 7,
	BOAT = 8,
	TRAFFIC_LIGHT = 9,
	FIRE_HYDRANT = 10,
	STOP_SIGN = 11,
	PARKING_METER = 12,
	BENCH = 13,
	BIRD = 14,
	CAT = 15,
	DOG = 16,
	HORSE = 17,
	SHEEP = 18,
	COW = 19,
	ELEPHANT = 20,
	BEAR = 21,
	ZEBRA = 22,
	GIRAFFE = 23,
	BACKPACK = 24,
	UMBRELLA = 25,
	HANDBAG = 26,
	TIE = 27,
	SUITCASE = 28,
	FRISBEE = 29,
	SKIS = 30,
	SNOWBOARD = 31,
	SPORTS_BALL = 32,
	KITE = 33,
	BASEBALL_BAT = 34,
	BASEBALL_GLOVE = 35,
	SKATEBOARD = 36,
	SURFBOARD = 37,
	TENNIS_RACKET = 38,
	BOTTLE = 39,
	WINE_GLASS = 40,
	CUP = 41,
	FORK = 42,
	KNIFE = 43,
	SPOON = 44,
	BOWL = 45,
	BANANA = 46,
	APPLE = 47,
	SANDWICH = 48,
	ORANGE = 49,
	BROCCOLI = 50,
	CARROT = 51,
	HOT_DOG = 52,
	PIZZA = 53,
	DONUT = 54,
	CAKE = 55,
	CHAIR = 56,
	SOFA = 57,
	POTTEDPLAT = 58,
	BED = 59,
	DININGTABLE = 60,
	TOILET = 61,
	TVMONITOR = 62,
	LAPTOP = 63,
	MOUSE = 64,
	REMOTE = 65,
	KEYBOARD = 66,
	CELL_PHONE = 67,
	MICROWAVE = 68,
	OVEN = 69,
	TOASTER = 70,
	SINK = 71,
	REFRIGERATOR = 72,
	BOOK = 73,
	CLOCK = 74,
	VASE = 75,
	SCISSORS = 76,
	TEDDY_BEAR = 77,
	HAIR_DRIER = 78,
	TOOTHBRUSH = 79
};
static const std::map<coco_label, std::string> labelname = {
	{ coco_label::PERSON , "PERSON"},
	{ coco_label::BICYCLE , "BICYCLE"},
	{ coco_label::CAR , "CAR"},
	{ coco_label::MOTORBIKE , "MOTORBIKE"},
	{ coco_label::AEROPLANE , "AEROPLANE"},
	{ coco_label::BUS , "BUS"},
	{ coco_label::TRAIN , "TRAIN"},
	{ coco_label::TRUCK , "TRUCK"},
	{ coco_label::BOAT , "BOAT"},
	{ coco_label::TRAFFIC_LIGHT , "TRAFFIC_LIGHT"},
	{ coco_label::FIRE_HYDRANT , "FIRE_HYDRANT"},
	{ coco_label::STOP_SIGN , "STOP_SIGN"},
	{ coco_label::PARKING_METER , "PARKING_METER"},
	{ coco_label::BENCH , "BENCH"},
	{ coco_label::BIRD , "BIRD"},
	{ coco_label::CAT , "CAT"},
	{ coco_label::DOG , "DOG"},
	{ coco_label::HORSE , "HORSE"},
	{ coco_label::SHEEP , "SHEEP"},
	{ coco_label::COW , "COW"},
	{ coco_label::ELEPHANT , "ELEPHANT"},
	{ coco_label::BEAR , "BEAR"},
	{ coco_label::ZEBRA , "ZEBRA"},
	{ coco_label::GIRAFFE , "GIRAFFE"},
	{ coco_label::BACKPACK , "BACKPACK"},
	{ coco_label::UMBRELLA , "UMBRELLA"},
	{ coco_label::HANDBAG , "HANDBAG"},
	{ coco_label::TIE , "TIE"},
	{ coco_label::SUITCASE , "SUITCASE"},
	{ coco_label::FRISBEE , "FRISBEE"},
	{ coco_label::SKIS , "SKIS"},
	{ coco_label::SNOWBOARD , "SNOWBOARD"},
	{ coco_label::SPORTS_BALL , "SPORTS_BALL"},
	{ coco_label::KITE , "KITE"},
	{ coco_label::BASEBALL_BAT , "BASEBALL_BAT"},
	{ coco_label::BASEBALL_GLOVE , "BASEBALL_GLOVE"},
	{ coco_label::SKATEBOARD , "SKATEBOARD"},
	{ coco_label::SURFBOARD , "SURFBOARD"},
	{ coco_label::TENNIS_RACKET , "TENNIS_RACKET"},
	{ coco_label::BOTTLE , "BOTTLE"},
	{ coco_label::WINE_GLASS , "WINE_GLASS"},
	{ coco_label::CUP, "CUP"},
	{ coco_label::FORK, "FORK"},
	{ coco_label::KNIFE, "KNIFE"},
	{ coco_label::SPOON, "SPOON"},
	{ coco_label::BOWL, "BOWL"},
	{ coco_label::BANANA, "BANANA"},
	{ coco_label::APPLE, "APPLE"},
	{ coco_label::SANDWICH, "SANDWICH"},
	{ coco_label::ORANGE, "ORANGE"},
	{ coco_label::BROCCOLI, "BROCCOLI"},
	{ coco_label::CARROT, "CARROT"},
	{ coco_label::HOT_DOG, "HOT_DOG"},
	{ coco_label::PIZZA, "PIZZA"},
	{ coco_label::DONUT, "DONUT"},
	{ coco_label::CAKE, "CAKE"},
	{ coco_label::CHAIR, "CHAIR"},
	{ coco_label::SOFA, "SOFA"},
	{ coco_label::POTTEDPLAT, "POTTEDPLAT"},
	{ coco_label::BED, "BED"},
	{ coco_label::DININGTABLE, "DININGTABLE"},
	{ coco_label::TOILET, "TOILET"},
	{ coco_label::TVMONITOR, "TVMONITOR"},
	{ coco_label::LAPTOP, "LAPTOP"},
	{ coco_label::MOUSE, "MOUSE"},
	{ coco_label::REMOTE, "REMOTE"},
	{ coco_label::KEYBOARD, "KEYBOARD"},
	{ coco_label::CELL_PHONE, "CELL_PHONE"},
	{ coco_label::MICROWAVE, "MICROWAVE"},
	{ coco_label::OVEN, "OVEN"},
	{ coco_label::TOASTER, "TOASTER"},
	{ coco_label::SINK, "SINK"},
	{ coco_label::REFRIGERATOR, "REFRIGERATOR"},
	{ coco_label::BOOK, "BOOK"},
	{ coco_label::CLOCK, "CLOCK"},
	{ coco_label::VASE, "VASE"},
	{ coco_label::SCISSORS, "SCISSORS"},
	{ coco_label::TEDDY_BEAR, "TEDDY_BEAR"},
	{ coco_label::HAIR_DRIER, "HAIR_DRIER"},
	{ coco_label::TOOTHBRUSH, "TOOTHBRUSH"}
};

std::string coco_label_tostring(coco_label label);

class bbox {
public:
	bbox(float x, float y, float w, float h, float original_w, float original_h, float confidence, coco_label class_id, float padding = 1.25) {
		this->x = x;
		this->y = y;
		this->w = w;
		this->h = h;
		this->original_w = original_w;
		this->original_h = original_h;
		this->confidence = confidence;
		this->class_id = class_id;
		center = cv::Point(x + w / 2, y + h / 2);
		scale = cv::Point(w * padding, h * padding);
	}
	int x, y, w, h, original_w, original_h;
	float confidence;
	cv::Point center;
	cv::Point scale;
	coco_label class_id;
};
