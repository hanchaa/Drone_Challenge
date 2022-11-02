python ros_track.py \
--yolo-weights task2/yolov7/params/10class_800/best.pt \
--config-strongsort task2/strong_sort/configs/strong_sort.yaml \
--strong-sort-weights task2/osnet_x0_25_msmt17.pt \
--show-vid \
--conf-thres 0.60 \
--device 0
