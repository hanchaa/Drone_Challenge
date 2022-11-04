import os
import time
import json
import rospy
from urllib import request
from mavros_msgs.msg import State
from geometry_msgs.msg import PoseStamped

from ros_track import Rony2

rospy.init_node('opencv_example', anonymous=True)
rate = rospy.Rate(5)
rospy.loginfo("SUBSCRIBING START")

MESSAGE_MISSION_START = {
        "team_id": "user30",
        "command": "MISSION_START"
        }

api_url_mission = os.environ['REST_MISSION_URL']
api_url_answer = os.environ['REST_ANSWER_URL']
mission_trigger=True
counter_z=0

class MissionStart:
    def __init__(self):
        self.armed_state=False
        self.pose_z=0.

    def cb_state(self, msg):
        self.armed_state = msg.armed

    def cb_pose(self, msg):
        self.pose_z = msg.pose.position.z
    
    def posez_counter(self):
        global counter_z
        while counter_z < 20:
            if self.pose_z > 0.5:
                counter_z+=1
                time.sleep(0.2)
                
    def mission_start(self):
        global mission_trigger
        if mission_trigger==True:
            if self.armed_state == True and counter_z == 20:
                print("Take-Off and Mission Start!")
                data_mission = json.dumps(MESSAGE_MISSION_START).encode('utf8')
                print(data_mission)
                # req = request.Request(api_url_mission, data=data_mission)
                # resp = request.urlopen(req)
                # status = resp.read().decode('utf8')
                # if "OK" in status:
                #     print("Complete send : Mission Start!!")
                mission_trigger=False

def main():
    data_path = r'/home/agc2022/dataset/'
    data_list = os.listdir(data_path)
    data_list.sort()
    print(f"Get resque text & image.. {data_list}")

    rony2_model = Rony2()

    callbacks = MissionStart()
    sub_state=rospy.Subscriber("/scout/mavros/state", State, callbacks.cb_state)
    sub_pose=rospy.Subscriber("/scout/mavros/local_position/pose", PoseStamped, callbacks.cb_pose)
    callbacks.posez_counter()
    global mission_trigger
    while mission_trigger == True:
        callbacks.mission_start()

    ## TODO : 답안지 생성 & 제출 ##
    rony2_model()
    

if __name__ == "__main__":
    main()
