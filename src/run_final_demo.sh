# ran @ 163.152.26.113
#!/bin/bash

# source /opt/ros/melodic/setup.bash

# sh ./task2_audio/receiver.sh &

CUDA_VISIBLE_DEVICES=0 python demo_all.py \
--clue_path dataset \
--yolo_path assets/task2_final.pt \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights assets/osnet_x0_25_msmt17.pt \
--yolo_weights assets/task2_final.pt \
--max_confidence 0.3 \
--video_path videos/set05_drone03.mp4 \
--show_vid 