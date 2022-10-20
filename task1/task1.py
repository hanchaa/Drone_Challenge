import time
import cv2
import argparse
import glob
import json
import apriltag
import numpy as np
from .utils.common_utils import MatchImageSizeTo, save_tag_match_imgs
from .utils.json_utils import json_postprocess
from .utils.args_utils import parse_args
from .model import img_model, txt_model


class Task1:
    def __init__(self, args):

        self.debug_path = args.debug_path
        self.img_conf_th = args.img_conf_th
        self.img_kp_th = args.img_kp_th
        # -----------------------------------------
        # image clue preprocessing
        # -----------------------------------------
        img_list = glob.glob(args.clue_path+'/*.png', recursive=True)
        img_list.sort()
        imgs=[]
        img_resizer = MatchImageSizeTo()
        for img_ in img_list:
            img = cv2.imread(img_, cv2.IMREAD_GRAYSCALE)
            img = img_resizer(img)
            imgs.append(img)
        self.img_clue = imgs
        
        # # -----------------------------------------
        # # text clue preprocessing
        # # -----------------------------------------
        # txts = glob.glob(args.clue_path+'/*.json', recursive=True)
        # txts.sort()
        # self.txts = txts

    def __call__(self, img: np.ndarray):
        result_im = img_model(img, self.img_clue, self.debug_path, self.img_conf_th, self.img_kp_th)

        images = []
        detections = []
        tag_id = []
        # for i in range(0, len(result_im)):
        #     images.append(cv2.cvtColor(result_im[i][0], cv2.COLOR_BGR2GRAY))
        #     detector = apriltag.Detector()
        #     detections.append(detector.detect(images[i]))
        #     tag_id = []
        #     if len(detections[i]) == 0:
        #         continue
        #     else:
        #         for j in range(0, len(detections)):
        #             if len(detections[j]) == 0:
        #                 continue
        #             else:
        #                 for k in range(0, len(detections[j])):
        #                     tag_id.append(detections[j][k].tag_id)

        tag_id_set = set(tag_id)
        tag_id_list = list(tag_id_set)
        clue_num = [1]
        data = [clue_num[0], tag_id_list]

        # -----------------------------------------
        # json export
        # -----------------------------------------
        num_clues = len(clue_num)
        json_output = json_postprocess(num_clues, data)
        # with open(args.output_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_output, f, indent=4)
        
        print(json.dumps(json_output, ensure_ascii=False, indent=4))
        # print("TIME :", time.time()-start)

if __name__ == "__main__":
    start = time.time()
    args = parse_args()
    task1 = Task1(**vars(args))
    
    # -----------------------------------------
    # video input preprocessing 
    # TODO: real-time video input handling
    # -----------------------------------------
    # frames = []
    # cap = cv2.VideoCapture(args.video_path)
    # while (cap.isOpened()):
    #     ret, frame = cap.read()
    #     frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #     if(type(frame) == type(None)):
    #         break
    #     if frame_pos % args.frame_skip != 0:
    #         continue
    #     frames.append(frame)
    # cap.release()
    frames = '/home/eulrang/workspace/git/AGC2022_round3_task1/data_toy/rescue_image01.png'

    task1(frames)