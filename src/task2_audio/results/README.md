# for integrating drone challenge tasks

## video data & pretrained params
https://drive.google.com/drive/folders/1CfVMSrl-t0t2sMdNLMnU23oKJSOd1dQS?usp=sharing

## dependencies
``` commandline
pip install -r requirements.txt
```

## ros realtime inference command
``` shell
sh run.sh
```

## for debugging with video
```shell
python debug.py \
--video-path {PATH_TO_VIDEO}
--{ARGUMENTS FOR MODEL}
```
