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
    task = Task1(args)

    while True:
        retval, frame = cap.read()
        if not retval:
            break

        num_frames += 1
        state = num_frames // 1000
        
        with torch.no_grad():
            task(frame, state)

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()
