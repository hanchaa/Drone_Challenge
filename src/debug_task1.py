import torch
import cv2
import os

from tools import parse_args
from task1 import Task1

if __name__ == "__main__":
    args = parse_args()

    assert os.path.isfile(args.video_path), "No video exists!"

    cap = cv2.VideoCapture(args.video_path)
    del args.video_path
    
    num_frames = 0
    
    template = {
        "team_id": "mlvlab",
        "secret": "h8pnwElZ3FBnCwA4",
        "answer_sheet": {}
    }

    # task1 init
    task = Task1(args)

    prev_state = -1
    while True:
        retval, frame = cap.read()
        if not retval:
            break
        
        # 0: 상승후 첫 방 입장까지의 복도
        # 1: 1번 방 (90-960)
        # 2: 1번 방에서 나오고 복도 (961-1291)
        # 3: 2번방 (1292-1922)
        # 4: 2번 방에서 나오고 복도 (1923-2253)
        # 5: 3번 방 (2254-3064)
        # 6: 3번 방에서 나오고 착지 이전까지

        num_frames += 1
        internal_state = num_frames // 1000 

        
        if num_frames < 90:
            state = 0
        elif (num_frames <= 960 and num_frames >= 90):
            state = 1
        elif (num_frames <= 1291 and num_frames >= 961):
            state = 0
        elif (num_frames <= 1922 and num_frames >= 1292):
            state = 3
        elif (num_frames <= 2253 and num_frames >= 1923):
            state = 0
        elif (num_frames <= 3064 and num_frames >= 2254):
            state = 5
        elif num_frames >= 3065:
            state = 0
        else:
            state = 0
        
        with torch.no_grad():
            result_task1 = task(frame, state)

        if prev_state == 1 and state == 2:
            room_id = 1
            result_task1['answer_sheet']['room_id'] = room_id

        if prev_state == 3 and state == 4:
            room_id = 2
            result_task1['answer_sheet']['room_id'] = room_id

        if prev_state == 5 and state == 6:
            room_id = 3
            result_task1['answer_sheet']['room_id'] = room_id

        if prev_state == 6 and state == 7:
            MESSAGE_MISSION_END = {
                    "team_id": "mlvlab",
                    "secret": "h8pnwElZ3FBnCwA4",
                    "end_of_mission": "true"
            }
        
        prev_state = state
        print(f"num frames: {num_frames}, states: {state}, room id: {result_task1['answer_sheet']['room_id']}")

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()
