import torch
import cv2
import os
import json
from urllib import request


from tools import parse_args
# from task1 import Task1
from task2_vision import Task2Vision
# from task3 import Task3
import pandas as pd


if __name__ == "__main__":
    
    
    # load environment variable
    # api_url_answer = os.environ["REST_ANSWER_URL"]  
    # api_url_mission = os.environ["REST_MISSION_URL"]
    # data_path = '/home/agc2022/dataset'     
    
    args = parse_args()
    # task1 = Task1(args)
    task2_vision = Task2Vision(args)
    #task2_audio = TaskAudio(args)
    # task3 = Task3(**vars(args))

    # --------------------------------------
    # MISSION START
    # MESSAGE_MISSION_START = {
    #     "team_id": "mlvlab",
    #     "command": "MISSION_START"
    # }
    # print("Take-Off and Mission Start!")
    # data_mission = json.dumps(MESSAGE_MISSION_START).encode('utf8')
    # req = request.Request(api_url_mission, data=data_mission)
    # resp = request.urlopen(req)
    # status = resp.read().decode('utf8')
    # if "OK" in status:
    #     print("Complete send : Mission Start!!")
    # elif "ERROR" == status:    
    #     raise ValueError("Receive ERROR status. Please check your source code.")
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
    # framedf = pd.read_csv('state0301.csv')
    # framedf = pd.read_csv('state0302.csv')
    # framedf = pd.read_csv('state0303.csv')
    # framedf = pd.read_csv('state0501.csv') done
    # framedf = pd.read_csv('state0502.csv')
    framedf = pd.read_csv('state0503.csv')
    framecount = 0
    det_infos = []
    while True:
        retval, frame = cap.read()
        if not retval:
            break
        # if framecount < 200 :
        #     framecount += 1
        #     continue
        # if framecount > 800 :
        #     break
        state = framedf.iloc[framecount].state
        fid = framedf.iloc[framecount].frame[5:]
        framecount += 1

        with torch.no_grad():
            # import pdb; pdb.set_trace()
            
            # result_task1 = task1(frame, state)
            result_task2, detection_info = task2_vision(frame, state, fid=fid, cap=cap)
            print(fid, ':', result_task2['answer_sheet'])
            #task2_audio(frame,state)
            # result_task3 = task3(frame,state)


        prev_state = state

        # det_infos.append(detection_info)
        # if framecount > 300 :
        #     break
    # import pickle
    # with open('0301_300frames.pkl', 'wb') as f :
    #     pickle.dump(det_infos, f)

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()



# import torch
# import cv2
# import os

# from tools import parse_args
# from task2_vision import Task2Vision

# if __name__ == "__main__":
#     args = parse_args()

#     assert os.path.isfile(args.video_path), "No video exists!"

#     cap = cv2.VideoCapture(args.video_path)
#     del args.video_path

#     num_frames = 0
#     task = Task2Vision(args)

#     while True:
#         retval, frame = cap.read()
#         if not retval:
#             break

#         num_frames += 1
#         state = num_frames // 1000
        
#         with torch.no_grad():
#             task(frame, 1)

#     if cap.isOpened():
#         cap.release()

#     cv2.destroyAllWindows()