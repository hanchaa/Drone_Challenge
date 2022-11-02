#!/bin/bash

source /opt/ros/melodic/setup.bash

mkdir -p /home/catkin_ws/src/rospy_agc

cd /home/catkin_ws
catkin_make
cd /home/catkin_ws/src
catkin_create_pkg rospy_agc sensor_msgs mavros_msgs geometry_msgs rospy
source /home/catkin_ws/devel/setup.bash


cd /
source user_command.sh

exec "$@"
