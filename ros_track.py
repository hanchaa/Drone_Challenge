import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image

from tools import parse_args

from task1 import Task1
from task2 import Task2
# from task3 import Task3

class RosImageGetter:
    def __init__(self, args):
        self.task1 = Task1(args)
        print("Task 1 model is initialized!")

        self.task2 = Task2(args)
        print("Task 2 model is initialized!")

        # self.task3 = Task3()
        # print("Task 3 model is initialized!")

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

    def callback(self, data):
        image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            self.task1(image)
            self.task2(image)


if __name__ == "__main__":
    args = parse_args()
    del args.video_path

    getter = RosImageGetter(args)
    rospy.init_node("ros_image_getter", anonymous=True)
    rospy.spin()
