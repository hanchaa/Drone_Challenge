
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
import imgproc
import json
import zipfile
from task3_utils import *
from task3_parse_args import parse_args

from craft import CRAFT ## detection model
from wiw import WIW ## recognition model

import pdb
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="4"


class Task3:
    def __init__(self, **kwargs):
        ## Config
        self.text_threshold = kwargs['text_threshold']
        self.link_threshold = kwargs['link_threshold']
        self.low_text = kwargs['low_text']
        self.canvas_size = kwargs['canvas_size']
        self.mag_ratio = kwargs['mag_ratio']
        self.batch_max_length = kwargs['batch_max_length']


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

    def __call__(self, image):
        # image = intput_img[:, :, ::-1] ## # BGR to RGB
        image = np.ascontiguousarray(image)
    
        ### box inference
        bboxes, polys, score_text = inference(self.craft, image, self.text_threshold,\
                                    self.link_threshold, self.low_text, self.canvas_size, self.mag_ratio, False, None)
        
        image_show = image.copy()
        crop_image_list = []
        for bbox in bboxes:
            x_min = int(bbox[0][0]) 
            y_min = int(bbox[0][1])
            x_max = int(bbox[2][0])
            y_max = int(bbox[2][1])

            cv2.rectangle(image_show,(x_min, y_min),(x_max, y_max), (0, 255 , 0), 2)

            ## crop image processing ##
            cropped_image = image[y_min:y_max, x_min:x_max] ## [H,W,3]
            cropped_img = Image.fromarray(cropped_image)
            cropped_img = cropped_img.convert('L')

            cropped_img = self.transform(cropped_img).unsqueeze(0) ## [1, 1, H, W] 
            crop_image_list.append(cropped_img)

        ######### Recognition #########
        image_tensors = torch.cat(crop_image_list)
        batch_size = image_tensors.size(0)
        crop_image_input = image_tensors.cuda()

        length_for_pred = torch.IntTensor([self.batch_max_length] * batch_size).cuda()
        text_for_pred = torch.LongTensor(batch_size,self.batch_max_length + 1).fill_(0).cuda()

        preds = self.wiw(crop_image_input, text_for_pred, is_train=False)

        # select max probabilty (greedy decoding) then decode index to character
        _, preds_index = preds.max(2)
        preds_txt = self.converter.decode(preds_index, length_for_pred)


        ###### save png #######
        # cv2.imwrite('result/test.png', image_show)
        ###### show cv  ######
        # cv2.imshow('Visualize', image_show)
        # cv2.waitKey(1) 
        ##### print text #####
        for text in preds_txt:
            print(text) ## 작동하는지만 볼 것

        # -----------------------------------------
        # json export
        # -----------------------------------------
        # TODO: Text post processing 
        preds_answer = None
        num_texts = len(preds_txt)
        if num_texts == 0:
            preds_answer = "UNCLEAR"
        else:
            preds_answer = preds_txt[0]
  
        json_output = json_postprocess(preds_answer)
        # with open(args.output_path, 'w', encoding='utf-8') as f:
        #     json.dump(json_output, f, indent=4)
        
        print(json.dumps(json_output, ensure_ascii=False, indent=4))

if __name__ == "__main__":
    start = time.time()
    args = parse_args()
    task3 = Task3(**vars(args))


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


    image_path = './Image_example/간판들.png'
    with open(image_path, 'rb') as f:
        data = f.read()
    encoded_img = np.fromstring(data, dtype = np.uint8)
    image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) ## (H,W,3) BGR

    # pdb.set_trace()
    with torch.no_grad():
        task3(image)

    print("TASK3 TIME :", time.time()-start)
    