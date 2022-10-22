# for drone challenge task2
original cloned repository: https://github.com/mikel-brostrom/Yolov7_StrongSORT_OSNet

## video data & pretrained params
https://drive.google.com/drive/folders/1CfVMSrl-t0t2sMdNLMnU23oKJSOd1dQS?usp=sharing

## video inference command

``` shell
python task1.py \
--yolo-weights yolov7/params/5class_2000/best.pt \
--strong-sort-weights osnet_x0_25_msmt17.pt \
--source yolov7/video/set05_drone03.mp4 \
--save-vid \
--conf-thres 0.60 \
--device 0 \
--config-strongsort strong_sort/configs/strong_sort.yaml
```