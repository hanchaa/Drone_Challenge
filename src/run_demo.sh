#!/bin/bash

# source /opt/ros/melodic/setup.bash

# sh ./task2_audio/receiver.sh &

python demo_all.py \
--clue_path dataset \
--yolo_path task2_vision/weights/task2_final.pt \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_final.pt \
--max_confidence 0.3 \
--video_path task2_vision/yolov7/video/set05_drone01.mp4 \
--show_vid \
--device 0,1