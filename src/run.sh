#!/bin/bash

source /opt/ros/melodic/setup.bash

python main.py \
--yolo_path task1/weights/yolo.pt \
--clue_path dataset \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/yolov7_drone_dummy.pt \
--video_path task2_vision/yolov7/video/set03_drone03.mp4 \
--device 0