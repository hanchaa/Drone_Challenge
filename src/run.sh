#!/bin/bash

source /opt/ros/melodic/setup.bash

sh ./task2_audio/receiver.sh &

python main.py \
--clue_path dataset \
--yolo_path task1/weights/task1_safe_v1.pt \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe.pt \
--checkpoint ./task2_audio/ckpts/2022_11_03_17_47_47.0100.pt \
--max_confidence 0.6