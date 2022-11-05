# for drone challenge task2
original cloned repository: https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet

## video data & pretrained params
https://drive.google.com/drive/folders/1CfVMSrl-t0t2sMdNLMnU23oKJSOd1dQS?usp=sharing

## video inference command

<!-- ``` shell
python task1.py \
--yolo-weights yolov7/params/5class_2000/best.pt \
--strong-sort-weights osnet_x0_25_msmt17.pt \
--source yolov7/video/set05_drone03.mp4 \
--save-vid \
--conf-thres 0.60 \
--device 0 \
--config-strongsort strong_sort/configs/strong_sort.yaml
``` -->

Task2 Inference @ "src/"
```shell
python debug_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/drone_v1.pt \
--video_path task2_vision/yolov7/video/set03_drone03.mp4 \
--device 0
```

Train YOLOV7 @ "yolov7/"
```
python -m torch.distributed.launch --nproc_per_node 4 \
    train.py --epochs 1000  \
             --data data/vg.yaml \
             --weights ../weights/drone_v1.pt \
             --batch 16 \
             --device 0,1,6,7
```