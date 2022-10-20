import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import Image

from tools import parse_args
from task2 import Task2


class RosImageGetter:
    def __init__(self, args):
        del args.source

        self.task2 = Task2(**vars(args))
        print("Task 2 model is initialized!")

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.callback)

    def callback(self, data):
        image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with torch.no_grad():
            self.task2(image)


if __name__ == "__main__":
    args = parse_args()

    rospy.init_node("ros_image_getter", anonymous=True)
    getter = RosImageGetter(args)
    rospy.spin()
