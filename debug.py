import cv2
import os

from tools import parse_args
from task2 import Task2

if __name__ == "__main__":
    args = parse_args()

    assert os.path.isfile(args.video_path), "No video exists!"

    cap = cv2.VideoCapture(args.video_path)
    del args.video_path

    task = Task2(args)

    while True:
        retval, frame = cap.read()
        if not retval:
            break

        task(frame)

    if cap.isOpened():
        cap.release()

    cv2.destroyAllWindows()
