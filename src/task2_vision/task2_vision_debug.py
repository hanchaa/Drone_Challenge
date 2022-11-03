import torch
import cv2
import os

from task2_vision import Task2Vision

if __name__ == "__main__":
    from tools.parse_args import parse_args
    args = parse_args()

    assert os.path.isfile(args.video_path), "No video exists!"

    cap = cv2.VideoCapture(args.video_path)
    del args.video_path

    num_frames = 0
    task = Task2Vision(args)
    
    while True:
        retval, frame = cap.read()
        if not retval:
            break
        print(retval)

        num_frames += 1
        state = num_frames // 1000
        
        with torch.no_grad():
            task(frame, state)

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()
