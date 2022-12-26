import rospy
import cv2
import torch
import numpy as np
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import PoseStamped
import os
import json
from tools import parse_args
from urllib import request

from task1 import Task1
from task2_vision import Task2Vision
from task2_audio import Task2Audio
from task3 import Task3

class Rony2:
    def __init__(self):
        args = parse_args()
        del args.video_path

        # -1: 이륙 (어떤 방이든 처음으로 들어갈 때 까지)
        # 0: 복도
        # 1~: 방 번호
        self.state = -1
        self.prev_state = -1

        self.url_mission = os.environ["REST_MISSION_URL"]
        self.url_answer = os.environ["REST_ANSWER_URL"]

        self.template = {
        "team_id": "mlvlab",
        "secret": "h8pnwElZ3FBnCwA4",
        "answer_sheet": {}
        }

        self.result_task1 = {}
        self.result_task2 = {}
        self.result_task3 = {}

        self.result_audio = {}

        self.mission_ended = False

        self.task1 = Task1(args)
        print("Task 1 model is initialized!")

        self.task2_vision = Task2Vision(args)
        print("Task 2 vision model is initialized!")

        self.task2_audio = Task2Audio(args)
        print("Task 2 audio model is initialized!")

        self.task3 = Task3(**vars(args))
        print("Task 3 model is initialized!")

        self.image_sub = rospy.Subscriber("/camera/color/image_raw/compressed", CompressedImage, self.image_callback, queue_size=1, buff_size=2**24)
        self.coord_sub = rospy.Subscriber("/scout/mavros/vision_pose/pose", PoseStamped, self.coord_callback)

    def image_callback(self, data):

        if self.prev_state > 0 and self.state == 0:
            try:
                audio_result = self.result_audio[self.prev_state]

                if self.result_task2['answer_sheet']['answer']["person_num"]["M"] != 'UNCLEAR':
                    self.result_task2['answer_sheet']['answer']['person_num']["M"] = str(
                        int(self.result_task2['answer_sheet']['answer']['person_num']["M"]) + audio_result["male"])

                if self.result_task2['answer_sheet']['answer']['person_num']["W"] != 'UNCLEAR':
                    self.result_task2['answer_sheet']['answer']['person_num']["W"] = str(
                        int(self.result_task2['answer_sheet']['answer']['person_num']["W"]) + audio_result["female"])

                if self.result_task2['answer_sheet']['answer']['person_num']["C"] != 'UNCLEAR':
                    self.result_task2['answer_sheet']['answer']['person_num']["C"] = str(
                        int(self.result_task2['answer_sheet']['answer']['person_num']["C"]) + audio_result["baby"])
            except:
                pass

            self.result_task2['answer_sheet']['room_id'] = self.result_task1["answer_sheet"]["room_id"]
            self.result_task3['answer_sheet']['room_id'] = self.result_task1["answer_sheet"]["room_id"]

            for i in range(1, 4):
                self.template['answer_sheet'] = eval(f"self.result_task{i}")
                data = json.dumps(self.template).encode('unicode-escape')
                req = request.Request(self.url_answer, data=data)
                resp = request.urlopen(req)
                status = resp.read().decode('utf8')
                if "OK" in status:
                    print("Complete send : Answersheet!!")
                elif "ERROR" == status:
                    raise ValueError("Receive ERROR status. Please check your source code.")

        image = np.fromstring(data.data, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        with torch.no_grad():
            try:
                self.result_task1 = self.task1(image.copy(), self.state)    # NOTE: return added
            except:
                self.result_task1 = {
                    "answer_sheet": {
                        "room_id": "500",
                        "mission": "1",
                        "answer": {
                            "person_id": "UNCLEAR"
                        }
                    }
                }

            try:
                self.result_task2, poster_bbox = self.task2_vision(image.copy(), self.state)

            except:
                self.result_task2 = {
                    "answer_sheet": {
                        "room_id": "500",
                        "mission": "2",
                        "answer": {
                            "person_num": {
                                "M": "UNCLEAR",
                                "W": "UNCLEAR",
                                "C": "UNCLEAR"
                            }
                        }
                    }
                }
                poster_bbox = None

            try:
                self.result_task3 = self.task3(image.copy(), self.state, poster_bbox)
            except:
                self.result_task3 = {
                    "answer_sheet": {
                        "room_id": "500",
                        "mission": "3",
                        "answer": {
                            "place": "UNCLEAR"
                        }
                    }
                }

        self.prev_state = self.state

    def coord_callback(self, data):
        x, y = data.pose.position.x, data.pose.position.y
        old_state = self.state

        # 1번 방
        if (old_state == 0 or old_state == -1) and float(x) < 0 and float(y) < -2.2 and float(y) > -7.7:
            self.state = 1

        # 2번 방
        if (old_state == 0 or old_state == -1) and float(x) < 0 and float(y) < -7.7 and float(y) > -14.2:
            self.state = 2

        # 3번 방
        if (old_state == 0 or old_state == -1) and float(x) < 0 and float(y) < -14.2 and float(y) > -20.75:
            self.state = 3

        # 4번 방
        if (old_state == 0 or old_state == -1) and float(x) < 0 and float(y) < 10 and float(y) > 3.5:
            self.state = 4

        # 5번 방
        if (old_state == 0 or old_state == -1) and float(x) > 3 and float(y) < 8.1 and float(y) > 4:
            self.state = 5

        # 복도
        if old_state == 1 and float(x) > 0.5:
            self.state = 0

        # 복도
        if old_state == 2 and float(x) > 0.5:
            self.state = 0

        # 복도
        if old_state == 3 and float(x) > 0.5:
            self.state = 0

        # 복도
        if old_state == 4 and float(x) > 0.5:
            self.state = 0

        # 복도
        if old_state == 5 and float(x) < 2.5:
            self.state = 0

        # 착륙 중
        if self.mission_ended == False and old_state == 0 and data.pose.position.z < 0.5 :
            self.mission_ended = True
            # request end of mission message
            MESSAGE_MISSION_END = {
                "team_id": "mlvlab",
                "secret": "h8pnwElZ3FBnCwA4",
                "end_of_mission": "true"
            }
            data_mission = json.dumps(MESSAGE_MISSION_END).encode('utf8')
            req = request.Request(self.url_answer, data=data_mission)
            resp = request.urlopen(req)
            status = resp.read().decode('utf8')
            if "OK" in status:
                print("Complete send : Mission END!!")
            elif "ERROR" == status:
                raise ValueError("Receive ERROR status. Please check your source code.")

        if old_state != self.state:
            print(f"state is changed from {old_state} to {self.state}")

    def __call__(self):
        while not rospy.is_shutdown():
            with torch.no_grad():
                self.result_audio[self.state] = self.task2_audio(self.state)


if __name__ == "__main__":
    rospy.init_node("ros_node")
    model = Rony2()
    model()
