import torch
import cv2
import os
import json
from urllib import request


from tools import parse_args
from task1 import Task1
from task2_vision import Task2Vision
from task2_audio import Task2Audio
from task3 import Task3


if __name__ == "__main__":
    
    # 0: 상승후 첫 방 입장까지의 복도
    # 1: 1번 방
    # 2: 1번 방에서 나오고 복도
    # 3: 2번방
    # 4: 2번 방에서 나오고 복도
    # 5: 3번 방
    # 6: 3번 방에서 나오고 착지 이전까지
    
    # load environment variable
    api_url_answer = os.environ["REST_ANSWER_URL"]  
    api_url_mission = os.environ["REST_MISSION_URL"]
    data_path = '/home/agc2022/dataset'     
    
    args = parse_args()
    task1 = Task1(args)
    task2_vision = Task2Vision(args)
    #task2_audio = TaskAudio(args)
    task3 = Task3(**vars(args))

    # --------------------------------------
    # MISSION START
    MESSAGE_MISSION_START = {
        "team_id": "mlvlab",
        "command": "MISSION_START"
    }
    print("Take-Off and Mission Start!")
    data_mission = json.dumps(MESSAGE_MISSION_START).encode('utf8')
    req = request.Request(api_url_mission, data=data_mission)
    resp = request.urlopen(req)
    status = resp.read().decode('utf8')
    if "OK" in status:
        print("Complete send : Mission Start!!")
    elif "ERROR" == status:    
        raise ValueError("Receive ERROR status. Please check your source code.")
    # ----------------------------------------

    assert os.path.isfile(args.video_path), "No video exists!"

    cap = cv2.VideoCapture(args.video_path)
    del args.video_path

    num_frames = 0
    
    template = {
        "team_id": "mlvlab",
        "secret": "h8pnwElZ3FBnCwA4",
        "answer_sheet": {}
    }
    
    prev_state = -1
    while True:
        retval, frame = cap.read()
        if not retval:
            break

        num_frames += 1
        state = num_frames // 10 
        print(f"num frames: {num_frames}")

        with torch.no_grad():
            # import pdb; pdb.set_trace()
            
            result_task1 = task1(frame, state)
            result_task2 = task2_vision(frame,state)
            #task2_audio(frame,state)
            result_task3 = task3(frame,state)

        if prev_state == 1 and state == 2 :
            room_id = 1
            result_task1['answer_sheet']['room_id'] = room_id
            result_task2['answer_sheet']['room_id'] = room_id
            result_task3['answer_sheet']['room_id'] = room_id

            for i in range(1,4):
                template['answer_sheet'] = eval(f"result_task{i}")
                data = json.dumps(template).encode('unicode-escape')
                req = request.Request(api_url_answer,data=data)
                resp = request.urlopen(req)
                status = resp.read().decode('utf8')
                if "OK" in status:
                    print("Complete send : Answersheet!!")
                elif "ERROR" == status:    
                    raise ValueError("Receive ERROR status. Please check your source code.")

                

        if prev_state == 3 and state == 4 :
            room_id = 2
            result_task1['answer_sheet']['room_id'] = room_id
            result_task2['answer_sheet']['room_id'] = room_id
            result_task3['answer_sheet']['room_id'] = room_id

            for i in range(1,4):
                template['answer_sheet'] = eval(f"result_task{i}")
                data = json.dumps(template).encode('unicode-escape')
                req = request.Request(api_url_answer,data=data)
                resp = request.urlopen(req)
                status = resp.read().decode('utf8')
                if "OK" in status:
                    print("Complete send : Answersheet!!")
                elif "ERROR" == status:    
                    raise ValueError("Receive ERROR status. Please check your source code.")



        if prev_state == 5 and state == 6 :
            room_id = 3
            result_task1['answer_sheet']['room_id'] = room_id
            result_task2['answer_sheet']['room_id'] = room_id
            result_task3['answer_sheet']['room_id'] = room_id

            for i in range(1,4):
                template['answer_sheet'] = eval(f"result_task{i}")
                data = json.dumps(template).encode('unicode-escape')
                req = request.Request(api_url_answer,data=data)
                resp = request.urlopen(req)
                status = resp.read().decode('utf8')
                if "OK" in status:
                    print("Complete send : Answersheet!!")
                elif "ERROR" == status:    
                    raise ValueError("Receive ERROR status. Please check your source code.")

        if prev_state == 6 and state == 7:
            MESSAGE_MISSION_END = {
                    "team_id": "mlvlab",
                    "secret": "h8pnwElZ3FBnCwA4",
                    "end_of_mission": "true"
                }
            data_mission = json.dumps(MESSAGE_MISSION_END).encode('utf8')
            req = request.Request(api_url_answer, data=data_mission)
            resp = request.urlopen(req)
            status = resp.read().decode('utf8')
            if "OK" in status:
                print("Complete send : Mission End!!")
            elif "ERROR" == status:    
                raise ValueError("Receive ERROR status. Please check your source code.")
            break

        prev_state = state

            
    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()

