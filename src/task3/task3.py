
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data
import torch.nn.functional as F

from PIL import Image

import cv2
from skimage import io
import numpy as np
import string
import json
import zipfile
from .task3_utils import *
from .task3_parse_args import parse_args
from .imgproc import *
from .craft import CRAFT ## detection model
from .wiw import WIW ## recognition model

import pdb

def inter(box1, box2, margin=20):
    # box = (x1, y1, x2, y2)
    # obtain x1, y1, x2, y2 of the intersection
    x1 = max(box1[0]-margin, box2[0]-margin)
    y1 = max(box1[1]-margin , box2[1]-margin)
    x2 = min(box1[2]+margin, box2[2]+margin)
    y2 = min(box1[3]+margin, box2[3]+margin)

    # compute the width and height of the intersection
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)

    inter = w * h
    return inter

class Task3:
    def __init__(self, **kwargs):
        # self.out_path = kwargs['out_path']
        ## Config
        # print(kwargs)
        ## state, confidence score, prediction text
        self.answer_list  = [[] for i in range(100)] ## len =3 ##각 인덱스가 state ##[confidence core, preds text]
        self.answer_cnt = [0 for i in range(100)]
        self.last_answer = "UNCLEAR"
        self.last_state = -1
        ######################
        self.text_threshold = kwargs['text_threshold']
        self.link_threshold = kwargs['link_threshold']
        self.low_text = kwargs['low_text']
        self.canvas_size = kwargs['canvas_size']
        self.mag_ratio = kwargs['mag_ratio']
        self.batch_max_length = kwargs['batch_max_length']
        self.max_confidence = kwargs['max_confidence']


        ##### Detection model load (CRAFT) #####
        craft = CRAFT()
        craft.load_state_dict(copyStateDict(torch.load(kwargs['craft_weight'])))
        craft = craft.cuda()
        self.craft = torch.nn.DataParallel(craft)
        self.craft.eval()

        ##### Recognition model load (WIW) #####
        cudnn.benchmark = True
        cudnn.deterministic = True

        if 'CTC' in kwargs['Prediction']:
            self.converter = CTCLabelConverter(kwargs['character'])
        else:
            self.converter = AttnLabelConverter(kwargs['character'])

        kwargs['num_class'] = len(self.converter.character)

        if kwargs['rgb']:
            kwargs['input_channel'] = 3
    
        wiw = WIW(**kwargs)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.wiw = torch.nn.DataParallel(wiw).cuda()
        self.wiw.load_state_dict(torch.load(kwargs['wiw_wieght'], map_location=device))
        self.wiw.eval()

        ## other utils
        self.transform = ResizeNormalize((kwargs['imgW'], kwargs['imgH']))
        # transform = NormalizePAD((3, self.imgH, resized_max_w))
        self.img_cnt=0

    def __call__(self, image, state):

        if self.last_state != state:
            self.last_answer = "UNCLEAR"
            self.last_state = state
        
        #print("GPU :",torch.cuda.current_device())
        image = image[:, :, ::-1] ## # BGR to RGB
        image = np.ascontiguousarray(image)
        ### box inference
        
        bboxes, polys, score_text = inference(self.craft, image, self.text_threshold,\
                                    self.link_threshold, self.low_text, self.canvas_size, self.mag_ratio, False, None)
        
        image_show = image.copy()
        crop_image_list = []
        bbox_list = []

        for bbox in bboxes:
            
            x_max = int(np.max(bbox[:,0]))
            x_min = int(np.min(bbox[:,0]))

            y_max = int(np.max(bbox[:,1]))
            y_min = int(np.min(bbox[:,1]))

            cv2.rectangle(image_show,(x_min, y_min),(x_max, y_max), (0, 255 , 0), 2)

            ## crop image processing ##
            try:
                cropped_image = image[y_min:y_max, x_min:x_max] ## [H,W,3]
                cropped_img = Image.fromarray(cropped_image)
                # cropped_img = cropped_img.convert('L') ## no rgb
                cropped_img = cropped_img.convert('RGB') ## rgb information

                cropped_img = self.transform(cropped_img).unsqueeze(0) ## [1, 1, H, W] 
                crop_image_list.append(cropped_img)
                bbox_list.append((x_min, y_min, x_max, y_max))


            except:
                continue
        
        ############ 겹치는거 제거 ############
        ## crop_image_list
        del_idx = []
        new_bbox_list = []
        for i,current_box in enumerate(bbox_list):
            if i == len(bbox_list)-1: break
            for remain in bbox_list[i+1:]:
                if inter(current_box, remain)>0:
                    ##겹치는 case
                    del_idx.append(bbox_list.index(current_box))
                    del_idx.append(bbox_list.index(remain))

        new_crop_image_list = []
        for idx in range(len(crop_image_list)):
            if idx not in del_idx:
                new_crop_image_list.append(crop_image_list[idx])
                new_bbox_list.append(bbox_list[idx])
        ###################################


        ############ y축 아래버리기 ############
        ## crop_image_list
        final_crop_image = []
        for idx, bbox in enumerate(new_bbox_list):
            if bbox[3] < image.shape[0]//2:
                final_crop_image.append(new_crop_image_list[idx])
        ###################################

        ######### Recognition #########
        confidence_score_list = []
        if len(final_crop_image)!=0:
            image_tensors = torch.cat(final_crop_image)
            batch_size = image_tensors.size(0)
            crop_image_input = image_tensors.cuda()

            length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).cuda()
            text_for_pred = torch.LongTensor(batch_size,self.batch_max_length + 1).fill_(0).cuda()
            
            preds = self.wiw(crop_image_input, text_for_pred, is_train=False)

            ## select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_txt = self.converter.decode(preds_index, length_for_pred)

            ## compute confidence score 
            preds_prob = F.softmax(preds, dim=2).detach().cpu()
            preds_max_prob, _ = preds_prob.max(dim=2)
            
            for pred, pred_max_prob in zip(preds_txt, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                confidence_score_list.append(confidence_score)

            ## final_text : confidence가 가장 높은 text
            final_answer = "UNCLEAR"

            max_confidence = max(confidence_score_list)
            if max_confidence > self.max_confidence:
                max_idx = confidence_score_list.index(max_confidence)
                final_answer = preds_txt[max_idx].split('[s]')[0]

                if 'x' in final_answer:
                    json_output = json_postprocess("UNCLEAR")
                    return json_output


                self.answer_list[state].append(final_answer)
            

            # -----------------------------------------
            # json export
            # -----------------------------------------
            # TODO: Text post processing 
            
            for state_idx in range(len(self.answer_list)):   ## 모든 state에 대해서
                pred_list = self.answer_list[state_idx]
                if (len(pred_list) > 0):
                    if state_idx!= state:
                        if (final_answer in pred_list):
                            self.answer_cnt[state] +=1
                            self.last_answer = final_answer
                            json_output = json_postprocess(final_answer)
                            return json_output

                        for pred_text in pred_list:
                            ## final_answer vs pred_text
                            cur_len = len(final_answer)
                            pred_len = len(pred_text)
                            
                            if cur_len > pred_len:
                                long_str = final_answer
                                short_str = pred_text
                            else:
                                long_str = pred_text
                                short_str = final_answer

                            if short_str in long_str:
                                self.answer_cnt[state] +=1
                                self.last_answer = long_str
                                json_output = json_postprocess(long_str)
                                return json_output

            
        else:
            if len(self.answer_list[state])==0:
                json_output = json_postprocess("UNCLEAR")
                return json_output
           
            # elif self.answer_cnt[state] == 0:
            #     json_output = json_postprocess("UNCLEAR")
            #     return json_output

        if self.answer_cnt[state] == 0:
            json_output = json_postprocess("UNCLEAR")
            return json_output

        else:
           json_output = json_postprocess(self.last_answer)
           return json_output
                    
        
if __name__ == "__main__":
    import sys
    args = parse_args()
    task3 = Task3(**vars(args))

    start = time.time()
    ## log 생성
    # sys.stdout = open('./result/2022_flight7.txt', 'w')
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

    # image_path = '/home/sojin/Drone_Challenge/task3/Image_example/간판들.png'
    image_list = '/hub_data2/drone_2022sample/flight07'
    
    for image_ in sorted(os.listdir(image_list)):
        if image_ == '.DS_Store': continue

        image_path = os.path.join(image_list, image_)
        print("------------------------------------ {} ------------------------------------".format(image_))
        with open(image_path, 'rb') as f:
            data = f.read()
        encoded_img = np.fromstring(data, dtype = np.uint8)
        image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) ## (H,W,3) BGR
        with torch.no_grad():
            answer_dict = task3(image)

        print(answer_dict)
    print("TASK3 TIME :", time.time()-start)
    # sys.stdout.close()
    
