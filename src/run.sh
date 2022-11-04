#!/bin/bash

source /opt/ros/melodic/setup.bash

sh ./task2_aduio/receiver.sh &

python main.py \
--yolo_path task1/weights/yolo.pt \
--clue_path task1/toy_test/case01 \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe.pt \
--device 0