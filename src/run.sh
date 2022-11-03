#!/bin/bash

source /opt/ros/melodic/setup.bash

python main.py \
--yolo_path task1/weights/yolo.pt \
--clue_path dataset \
--yolo_weights task2_vision/yolov7/params/10class_800/best.pt \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/osnet_x0_25_msmt17.pt \
--show_vid \
--conf_thres 0.60 \
--device 0