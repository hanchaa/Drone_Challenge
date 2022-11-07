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
    # del args.video_path

    num_frames = 0
    
    template = {
        "team_id": "mlvlab",
        "secret": "h8pnwElZ3FBnCwA4",
        "answer_sheet": {}
    }
    
    # load frame info
    tmp = args.video_path.split('/')[-1][3:5] + args.video_path.split('/')[-1][-6:-4]
    framedf = pd.read_csv(f'state{tmp}.csv')
    
    # define video writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    save_path = args.video_path[:-4] + '_all.mp4'

    video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))

    prev_state = -1
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

        frame_for_vis = frame.copy()

        with torch.no_grad():
            # result_task1 = task1(frame.copy(), state, frame_for_vis)
            result_task2, data_for_task3 = task2_vision(frame.copy(), state, frame_for_vis)
            # print(fid, ':', result_task2['answer_sheet'])
            #task2_audio(frame,state)
            # result_task3 = task3(frame.copy(), state, data_for_task3, frame_for_vis)

        # VISUALIZE
        if args.show_vid:
            titletext = f'{fid}   #2   State {state}'
            cv2.putText(frame_for_vis, titletext, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
            

            video_writer.write(frame_for_vis)
            print('video write success - frame',fid, '   state',state)
            cv2.waitKey(1)  # 1 millisecond     

        ###############

        prev_state = state


    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()
