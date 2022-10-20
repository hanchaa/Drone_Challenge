# for integrating drone challenge tasks

## video data & pretrained params
https://drive.google.com/drive/folders/1CfVMSrl-t0t2sMdNLMnU23oKJSOd1dQS?usp=sharing

## video inference command

``` shell
python task1.py \
--yolo-weights yolov7/params/5class_2000/best.pt \
--config-strongsort strong_sort/configs/strong_sort.yaml \
--strong-sort-weights osnet_x0_25_msmt17.pt \
--show-vid \
--conf-thres 0.60 \
--device 0 \
```
