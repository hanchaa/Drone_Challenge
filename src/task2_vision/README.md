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

Task2 SAFE VERSION @ "src/"
```shell
python debug_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe.pt \
--video_path task2_vision/yolov7/video/set03_drone03.mp4 \
--device 0
```
```shell
python debug_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe_v1.2.pt \
--video_path task2_vision/yolov7/video/set03_drone01.mp4 \
--device 0
```
```shell
python demo_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe_v3.0.pt \
--video_path task2_vision/yolov7/video/set03_drone01.mp4 \
--device 0
```
python demo_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe_v1.2.pt \
--video_path task2_vision/yolov7/video/set05_drone02.mp4 \
--show_vid \
--device 0
python demo_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe_v1.2.pt \
--video_path task2_vision/yolov7/video/set05_drone03.mp4 \
--show_vid \
--device 0


Task2 UNCLEAR VERSION @ "src/"
```shell
python debug_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe.pt \
--video_path task2_vision/yolov7/video/set03_drone03.mp4 \
--unclear_thres 0 \
--device 0
```


Task2 UNCLEAR VERSION @ "src/"
```shell
python debug_task2.py \
--config_strongsort task2_vision/strong_sort/configs/strong_sort.yaml \
--strong_sort_weights task2_vision/weights/osnet_x0_25_msmt17.pt \
--yolo_weights task2_vision/weights/task2_safe.pt \
--video_path task2_vision/yolov7/video/set03_drone03.mp4 \
--unclear_thres 0 \
--device 0
```


Train YOLOV7 @ "yolov7/"
```
python -m torch.distributed.launch --nproc_per_node 4 \
    train.py --epochs 50  \
             --data data/vg.yaml \
             --weights runs/train/exp12/weights/last.pt \
             --batch 20 \
             --device 0,1,6,7
```

```
python train.py --epochs 50  \
             --data data/vg.yaml \
             --weights runs/train/exp12/weights/last.pt \
             --batch 4 \
             --device 5
```

```
python -m torch.distributed.launch --nproc_per_node 6 \
    train.py --epochs 80  \
             --data data/vg.yaml \
             --weights ../weights/yolov7.pt \
             --batch 36 \
             --device 1,2,3,5,6,7
```