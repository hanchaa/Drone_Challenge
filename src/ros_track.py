import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
import os
import csv
import time

from tools import parse_args

from task1 import Task1
from task2_vision import Task2Vision
from task2_audio import Task2Audio
from task3 import Task3

class Ros:
    def __init__(self, args):
        # 0: 상승후 첫 방 입장까지의 복도
        # 1: 1번 방
        # 2: 1번 방에서 나오고 복도
        # 3: 2번방
        # 4: 2번 방에서 나오고 복도
        # 5: 3번 방
        # 6: 3번 방에서 나오고 착지 이전까지
        self.state = -1

        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback)
        self.coord_sub = rospy.Subscriber("/scout/mavros/vision_pose/pose", PoseStamped, self.coord_callback)
        self.pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

        self.task1 = Task1(args)
        print("Task 1 model is initialized!")

        self.task2_vision = Task2Vision(args)
        print("Task 2 vision model is initialized!")

        self.task2_audio = Task2Audio(args, self.pub)
        print("Task 2 audio model is initialized!")

        self.task3 = Task3(**vars(args))
        print("Task 3 model is initialized!")

    def image_callback(self, data):
        image = np.fromstring(data.data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        with torch.no_grad():
            self.task1(image, self.state)
            self.task2(image, self.state)
            self.task3(image, self.state)

    def coord_callback(self, data):
        x, y = data.pose.position.x, data.pose.position.y
        old_state = self.state

        if float(x) < 0 and float(y) < -17:
            self.state = 1

        if old_state == 1 and float(x) > 0.5:
            self.state = 2

        if old_state == 2 and float(y) > -14.6 and float(x) < 0:
            self.state = 3

        if old_state == 3 and float(x) > 0.5:
            self.state = 4

        if old_staet == 4 and float(y) > -7.3 and float(x) < 0:
            self.state = 5

        if old_state == 5 and float(x) > 0.5:
            self.state = 6

        if old_state != self.state:
            print(f"state is changed from {old_state} to {self.state}")

    def __call__(self):
        while not rospy.is_shutdown():
            with torch.no_grad():
                self.task2_audio(self.state)


if __name__ == "__main__":
    args = parse_args()
    del args.video_path

    rospy.init_node("ros_node")
    getter = Ros(args)

    rospy.spin()
