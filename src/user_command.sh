#!/bin/bash

mkdir -p /home/agc2022/dataset

mv main.py /home/catkin_ws/src/rospy_agc
mv requirements.txt /home/catkin_ws/src/rospy_agc

cd /home/catkin_ws/src/rospy_agc

chmod +x main.py
python3 main.py

exec "$@"
