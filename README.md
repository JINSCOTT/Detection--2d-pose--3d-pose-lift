# detection-> 2d pose estimation -> 3d pose lift

This is a demo project performing detection, 2d pose estimation and 3d pose estimation.
Weight files are either downloaded or generated with torch dynamo.
note that the red circles are 2d joints, purple circles are 3d

## FLOW

1. Load models
2. open camera
3. main loop
    3.1 Detection
    3.2 2d pose estimation
    3.3 3d pose lift

## ML models

### YOLOv7

* Performs the detection

### RTMPose

* Performs 2d pose joint estimation
* Performs "once" on each person object
* is batch-able

### Motion-bert

* performs 2d to 3d joint point lift
* the result is put back onto the image, but you should use the generated result, 
this is only used to show the result is sensible
* the model need a 10 frame buffer, so there will be a 10 frame delay
* batch number is one

## Reference

* Detection YOLOv7
  * <https://github.com/WongKinYiu/yolov7>
* 2d pose estimation RTMPOSE
  * <https://github.com/open-mmlab/mmpose/tree/main/projects/rtmpose>
  * <https://github.com/Tau-J/rtmlib>
* 2d pose to 3d pose Motion Bert
  * <https://github.com/Walter0807/MotionBERT>
