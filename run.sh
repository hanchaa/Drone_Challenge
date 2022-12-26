#!/bin/bash

source /opt/ros/melodic/setup.bash

sh ./task2_audio/receiver.sh &

CUDA_VISIBLE_DEVICES=0 python main.py \
--clue_path /home/agc2022/dataset \
--yolo_path task1/weights/task2_final.pt \
--img_conf_th 0.6 \
--img_kp_th 150 \
--txt_th 3 \
--od_th 0.3 \
--total_th 0.6 \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_final.pt \
--checkpoint ./task2_audio/ckpts/2022_11_03_17_49_30.0110.pt \
--max_confidence 0.3 \
--wiw_wieght ./task3/trained_model/best_accuracy_lr5e-2.pth