import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped

from tools import parse_args

from task1 import Task1
from task2 import Task2
from task3 import Task3

class RosImageGetter:
    def __init__(self, args):
        self.task1 = Task1(args)
        print("Task 1 model is initialized!")

        self.task2 = Task2(args)
        print("Task 2 model is initialized!")

        self.task3 = Task3()
        print("Task 3 model is initialized!")

        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback)
        self.coord_sub = rospy.Subscriber("/scout/mavros/vision_pose/pose", PoseStamped, self.coord_callback)

    def image_callback(self, data):
        image = np.fromstring(data.data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        with torch.no_grad():
            self.task1(image)
            self.task2(image)
            self.task3(image)

    def coord_callback(self, data):
        #coord_img = np.zeros((200, 1000, 3))
        #cv2.putText(coord_img, f"x: {data.pose.position.x}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
        #cv2.putText(coord_img, f"y: {data.pose.position.y}", (10, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        #cv2.imshow("coord", coord_img)
        #cv2.waitKey(1)
        print(f"x: {data.pose.position.x}")
        print(f"y: {data.pose.position.y}")


if __name__ == "__main__":
    args = parse_args()
    del args.video_path

    rospy.init_node("ros_image_getter")
    getter = RosImageGetter(args)
    rospy.spin()
