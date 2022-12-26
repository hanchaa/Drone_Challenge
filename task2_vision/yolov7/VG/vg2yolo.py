import cv2
import os 
import json
import argparse
import numpy as np
#from sklearn.model_selection import train_test_split
from pathlib import Path
import random
import pandas as pd 
from sklearn.model_selection import train_test_split
#vg의 x,y는 tl_x, tl_y이므로 yolo의 bbox label 형식에 맞게 normalized된 c_x, c_y로 변경
#vg의 w,h를 yolo의 bbox label 형식에 맞게 normalized된 w, h로 변경 
def get_args_parser():
    parser = argparse.ArgumentParser('trasform to yolo labeling', add_help=False)
    parser.add_argument('--train_output_dir', default='/home/shkwon129/project/yolov7/dataset/visual_genome/train/labels/')
    parser.add_argument('--valid_output_dir', default='/home/shkwon129/project/yolov7/dataset/visual_genome/valid/labels/')
    parser.add_argument('--json_path', default='/home/shkwon129/project/yolov7/VG/VG_json/vg_label.json')
    parser.add_argument('--img_path', default='/home/shkwon129/project/yolov7/dataset/visual_genome/train/images')
    return parser

def read_json(json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data



#vg의 x,y는 tl_x, tl_y이므로 yolo의 bbox label 형식에 맞게 normalized된 c_x, c_y로 변경
#vg의 w,h를 yolo의 bbox label 형식에 맞게 normalized된 w, h로 변경 
def vgbox2yolobox(ori_img, object_dict):
    w,h,x,y = object_dict['w'], object_dict['h'], object_dict['x'], object_dict['y']
    w0, h0 = ori_img.shape[1], ori_img.shape[0]
    w, h, c_x, c_y = w/w0, h/h0, (x+w/2)/w0, (y+h/2)/h0
    return c_x, c_y, w, h

#이제 안쓰임
# def vglabel2yololabel(object_dict):
#     lb1 = object_dict['label_man'] 
#     lb2 = object_dict['label_child']  
#     lb3 = object_dict['label_safe'] 
#     lb4 = object_dict['label_person']    
#     return lb1, lb2, lb3, lb4

def main(args):
    object_dicts = read_json(args.json_path)
    df = pd.DataFrame(object_dicts)
    wb_img_index = []
    for i in range(len(df)):
        if df.iloc[i]['label'] == 14:
            wb_img_index.append(df.iloc[i]['img_id'])

    train_index = df['img_id']>2333715
    add_img_idx = [76, 107, 347, 374, 380, 386]
    df_train = df[train_index]
    df_test = df[~train_index]
    #import pdb; pdb.set_trace()
    for i in range(len(df)):
        if df.iloc[i]['img_id'] in add_img_idx :        
            df_train = pd.concat([df_train,pd.DataFrame(df.iloc[i]).transpose()],axis=0)
            df_test = df_test[~(df_test['img_id']==df.iloc[i]['img_id'])]
    train = []
    for i in range(len(df_train)):
        train_dict = df_train.iloc[i].to_dict()
        train.append(train_dict)
    test = []
    for i in range(len(df_test)):
        test_dict = df_test.iloc[i].to_dict()
        test.append(test_dict)
    key_lists = ['man','woman','child','monitor','cabinet','basket','box','trash bin','computer','laptop','bookshelf',
    'chair', 'printer','desk','whiteboard']
    num2key = {i:key for i, key in enumerate(key_lists)}
    count_dict_train = dict()
    count_dict_test = dict()
    for key in key_lists:
        count_dict_train[key] = 0
        count_dict_test[key] = 0
    for data in train:
        count_dict_train[num2key[data['label']]] += 1
    for data in test:
        count_dict_test[num2key[data['label']]] += 1

    # for object_dict in train:
    #     img_id = object_dict['img_id']
    #     file_path = args.img_path+'/'+str(img_id)+'.jpg'
    #     ori_img = cv2.imread(file_path)
    #     lb = object_dict['label']
    #     c_x, c_y, w, h = vgbox2yolobox(ori_img, object_dict)
    #     with open(args.train_output_dir+str(img_id)+'.txt', 'a') as f:
    #         f.write(f'{lb} {c_x} {c_y} {w} {h} \n')
    #         print(f'{args.train_output_dir+str(img_id)}.txt에 저장되었습니다.')
    
    for object_dict in test:
        img_id = object_dict['img_id']
        file_path = args.img_path+'/'+str(img_id)+'.jpg'
        ori_img = cv2.imread(file_path)
        lb = object_dict['label']
        c_x, c_y, w, h = vgbox2yolobox(ori_img, object_dict)
        with open(args.valid_output_dir+str(img_id)+'.txt', 'a') as f:
            f.write(f'{lb} {c_x} {c_y} {w} {h} \n')
            print(f'{args.valid_output_dir+str(img_id)}.txt에 저장되었습니다.')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.train_output_dir:
        Path(args.train_output_dir).mkdir(parents=True, exist_ok=True)
    if args.train_output_dir:
        Path(args.train_output_dir).mkdir(parents=True, exist_ok=True)
    # img_path = 'drone_dataset/drone_VG/images'
    # json_path = 'drone_dataset/drone_VG/labels/drone_vg.json'
    main(args)