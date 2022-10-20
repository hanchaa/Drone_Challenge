# for integrating drone challenge tasks

## video data & pretrained params
https://drive.google.com/drive/folders/1CfVMSrl-t0t2sMdNLMnU23oKJSOd1dQS?usp=sharing

## dependencies
``` commandline
pip install -r requirements.txt
```

## ros realtime inference command
``` shell
python ros_track.py \
--yolo-weights task2/yolov7/params/10class_800/best.pt \
--config-strongsort task2/strong_sort/configs/strong_sort.yaml \
--strong-sort-weights task2/osnet_x0_25_msmt17.pt \
--show-vid \
--conf-thres 0.60 \
--device 0 \
```

## for debugging with video
```shell
python debug.py \
--video-path {PATH_TO_VIDEO}
--{ARGUMENTS FOR MODEL}
```
