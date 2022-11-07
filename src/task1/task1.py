import cv2
import glob
import apriltag
import numpy as np
import matplotlib.cm as cm
from numpy import random
import torch

from .utils.json_utils import json_preprocess, json_postprocess
from .utils.args_utils import parse_args
from .utils.model_utils import matching, read_image, make_matching_plot, plot_one_box
from .superglue.superpoint import SuperPoint
from .superglue.superglue import SuperGlue

from .yolov7.models.experimental import attempt_load
from .yolov7.utils.datasets import letterbox, LoadImages
from .yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, cv2
from .yolov7.utils.torch_utils import TracedModel


CLASS_MAP = {'person': 0, 'monitor': 1, 'cabinet': 2, 'basket': 3, 'box': 4, 'trash bin': 5, 'computer': 6, 'laptop': 7, 'bookshelf': 8, 'chair': 9, 'printer': 10, 'desk': 11,
             'whiteboard': 12, 'banner': 13, 'mirror': 14, 'stairs': 15, 'toy': 16, 'fire extinguisher': 17, 'poster': 18, 'sink': 19, 'exercise tool': 20, 'speaker': 21,
             'up_occluded': 22, 'up_red': 23, 'up_orange': 24, 'up_yellow': 25, 'up_green': 26, 'up_blue': 27, 'up_purple': 28, 'up_white': 29, 'up_gray': 30, 'up_black': 31,
             'low_occluded': 32, 'low_red': 33, 'low_orange': 34, 'low_yellow': 35, 'low_green': 36, 'low_blue': 37, 'low_purple': 38, 'low_white': 39, 'low_gray': 40, 'low_black': 41,
             'person_man': 42, 'person_woman': 43, 'person_child': 44, 'others_lifeguard': 45, 'others_medic': 46}

class Task1:
    def __init__(self, args):
        self.clue_path = args.clue_path
        self.json_output_path = args.json_output_path
        self.task1_debug = args.task1_debug
        self.debug_output_path = args.debug_output_path
        self.img_conf_th = args.img_conf_th
        self.img_kp_th = args.img_kp_th
        self.txt_th = args.txt_th
        self.od_th = args.od_th
        self.total_th = args.total_th
        self.show_video = args.show_vid
        self.cnt = 0
        self.state = 0
        self.room_id = None
        self.json = {'answer_sheet': {
                        'room_id': None,
                        'mission': "1",
                        'answer': {
                            'person_id': {}
                            }   
                        }   
                    }
        self.json_list = []

        # -----------------------------------------
        # image matching model & preprocessing
        # -----------------------------------------
        self.img_config = {
            'superpoint': {
                'nms_radius': 4,
                'keypoint_threshold': 0.005,
                'max_keypoints': 1024
            },
            'superglue': {
                'weights': 'indoor',
                'sinkhorn_iterations': 20,
                'match_threshold': 0.2,
            }
        }
        self.superpoint = SuperPoint(self.img_config.get('superpoint', {})).eval().to('cuda')
        self.superglue = SuperGlue(self.img_config.get('superglue', {})).eval().to('cuda')
        self.match_batch_size = 1

        # -----------------------------------------
        # YOLO model & preprocessing
        # -----------------------------------------
        self.imgsz = (640, 640)
        self.half = True
        self.conf_th = 0.25
        self.iou_th = 0.45
        self.classes = None
        self.cls_agnostic_nms = False
        self.yolo_path = args.yolo_path
        yolo = attempt_load(self.yolo_path, map_location='cuda').eval()
        self.stride = int(yolo.stride.max())
        self.img_size = check_img_size(self.imgsz[0], s=self.stride)
        self.names = yolo.names
        self.color_list = ['OCC','RED','ORG','YLW','GRN','BLU','PRP','WHT','GRY','BLK']
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        # if self.half:
        #     self.yolo = TracedModel(yolo, 'cuda', self.img_size).half()
        # else:
        #     self.yolo = TracedModel(yolo, 'cuda', self.img_size)
        self.yolo = yolo.half()
        # self.true=1 # NOTE: dummy code for debugging

    def __call__(self, img: np.ndarray, state, frame_for_vis=None):
        try:
            clue_info = []
            if (state == 0 or state == -1):  # NOTE: 복도에서 json, room_id 초기화
                self.json = {'answer_sheet': {
                        'room_id': None,
                        'mission': "1",
                        'answer': {
                            'person_id': {}
                            }   
                        }   
                    }
                self.json_list = []
                self.room_id = None
            
            # -----------------------------------------
            # text clue preprocessing
            # -----------------------------------------
            clue_txts = glob.glob(self.clue_path+'/*.json', recursive=True)
            clue_txt_list = ([])
            if len(clue_txts) > 0:
                clue_txts.sort()
                for clue_txt_ in clue_txts:
                    clue_txt_key = []
                    clue_txt_dict = json_preprocess(clue_txt_)
                    clue_txts_ = list(clue_txt_dict.values())[0]
                    for i in range(0, len(clue_txts_)):
                        clue_txt_key.append(CLASS_MAP[clue_txts_[i]])
                    clue_txt_list.append(clue_txt_key)

            # -----------------------------------------
            # image clue preprocessing
            # -----------------------------------------
            clue_img_list = glob.glob(self.clue_path+'/*.jpg', recursive=True)
            clue_imgs = []
            clue_imgs_p = []
            clue_imgs_scales = []
            if len(clue_img_list) > 0:
                clue_img_list.sort()
                for clue_img_ in clue_img_list:
                    clue_img_ = cv2.imread(clue_img_, cv2.IMREAD_GRAYSCALE)
                    image1, inp1, scales1 = read_image(clue_img_, [640, 480], 'cuda')   # NOTE: clue image
                    clue_imgs.append(image1)
                    clue_imgs_p.append(inp1)
                    clue_imgs_scales.append(scales1)

            # -----------------------------------------
            # Superglue inference
            # -----------------------------------------
            if self.task1_debug:
                input_img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            else:
                input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                       # NOTE: copy된 frame image 사용
            image0, inp0, scales0 = read_image(input_img, [640, 480], 'cuda')           # NOTE: video frame image

            if len(clue_img_list) > 0:
                score_img = []
                for i in range(0, len(clue_img_list)):                                  # NOTE: 각 이미지 단서마다 kpts, mean confidence 저장
                    pred, matches, conf = matching({'image0': inp0, 'image1': clue_imgs_p[i]}, self.superpoint, self.superglue)
                    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                    valid = matches > -1
                    mkpts0 = kpts0[valid]
                    mkpts1 = kpts1[matches[valid]]
                    mconf = conf[valid]                                                 # NOTE: superpoint 개수
                    score_img.append((mkpts0.shape[0], mconf.mean()))

                    if (score_img[i][0] > self.img_kp_th and score_img[i][1] > self.img_conf_th):
                        im_detections = []
                        im_detector = apriltag.Detector()
                        im_detections.append(im_detector.detect(input_img))
                        im_tag_id = []

                        for j in range(0, len(im_detections[0])):
                            im_tag_id.append(im_detections[0][j].tag_id)
                        im_json_output = json_postprocess(clue_img_list[i][-6:-4], im_tag_id)
                        self.json_list.append(im_json_output)
                    
                        if self.debug_output_path != None:                              # NOTE: for debugging (superpoint > 50 & confidence > 0.5 일 때만 이미지 저장)
                            color = cm.jet(mconf)
                            label = ['SuperGlue',
                                     'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                                     'Matches: {}'.format(len(mkpts0)),]
                            k_thresh = self.img_config['superpoint']['keypoint_threshold']
                            m_thresh = self.img_config['superglue']['match_threshold']
                            small_text = ['Keypoint Threshold: {:.4f}'.format(k_thresh),
                                          'Match Threshold: {:.2f}'.format(m_thresh),]
                            make_matching_plot(image0, clue_imgs[i], mkpts0, mkpts1, color, label,
                                            self.debug_output_path+'frame'+str(self.cnt)+'_clue'+str(i), small_text)
                                            
                    clue_info.append(clue_img_list[i][-6:-4])

            # -----------------------------------------
            # YOLO inference
            # -----------------------------------------
            if len(clue_txt_list) > 0:
                score_txt = 0.0
                score_bbox = 0.0
                for i in range(0, len(clue_txt_list)):
                    self.yolo(torch.zeros(1, 3, self.img_size, self.img_size).to('cuda').type_as(next(self.yolo.parameters())))
                    if self.task1_debug:
                        load_img = LoadImages(img, img_size=self.imgsz, stride=self.stride)
                        _, yolo_img, im0s, _ = next(iter(load_img))
                    else:
                        im0s = img
                        # im0s = frame_for_vis    # NOTE: video frame image 사용
                        yolo_img = letterbox(im0s, self.img_size, stride=self.stride)[0]
                        yolo_img = yolo_img[:, :, ::-1].transpose(2, 0, 1)
                        yolo_img = np.ascontiguousarray(yolo_img)
                    yolo_img = torch.from_numpy(yolo_img).to('cuda')
                    yolo_img = yolo_img.half() if self.half else yolo_img.float()
                    yolo_img /= 255.0
                    if len(yolo_img.shape) == 3:
                        yolo_img = yolo_img.unsqueeze(0)

                    pred = self.yolo(yolo_img)
                    pred = non_max_suppression(pred[0], self.conf_th, self.iou_th, self.classes, self.cls_agnostic_nms, multi_label=False, return_attributes=True)[0]
                    pred[:, :4] = scale_coords(yolo_img.shape[2:], pred[:, :4], im0s.shape).round()

                    if len(pred) > 0:
                        # NOTE: poster 사람 제거
                        person_pred = pred[0][pred[0][5] == 0]
                        not_person_pred = pred[0][pred[0][5] != 0]
                        poster_pred = pred[0][pred[0][5] == 18]
                        if len(person_pred) != 0 :
                            new_person_pred = []
                            for pep in person_pred :
                                flag = False
                                for pop in poster_pred :
                                    person_loc = pep[:4]
                                    poster_loc = pop[:4]
                                    person_left = person_loc[0] - person_loc[2]/2
                                    person_right = person_loc[0] + person_loc[2]/2
                                    person_top = person_loc[1] - person_loc[3]/2
                                    person_bottom = person_loc[1] + person_loc[3]/2
                                    poster_left = poster_loc[0] - poster_loc[2]/2
                                    poster_right = poster_loc[0] + poster_loc[2]/2
                                    poster_top = poster_loc[1] - poster_loc[3]/2
                                    poster_bottom = poster_loc[1] + poster_loc[3]/2
                                    if (poster_left < person_left) and (poster_top < person_top) and \
                                            (poster_right > person_right) and (poster_bottom > person_bottom):
                                        # person is in poster
                                        flag = True
                                        break
                                    else :
                                        flag = False
                                if not flag :
                                    new_person_pred.append(pep)
                            person_pred = torch.stack(new_person_pred)
                            pred = [torch.cat([person_pred, not_person_pred])][0]
                            
                        # NOTE: pred[0] = [X, Y, W, H, cls_conf, cls, upper_conf, upper_cls, lower_conf, lower_cls, ppl_conf, ppl_cls, oth_conf, oth_cls]
                        # NOTE: other confidence and other class not used in task1
                        cls_match_num = 0.0
                        for j in range(0, len(clue_txt_list[i])):
                            for k in range(0, pred.shape[0]):                                           # NOTE: bbox 여러개 쳐진 경우
                                if (pred[k][5] == 0 and pred[k][4] >= 0.7):                             # NOTE: 사람인경우
                                    if pred[k][11] == 0:
                                        name = 43
                                    elif pred[k][11] == 1:
                                        name = 43
                                    else:
                                        name = 44
                                    if name == clue_txt_list[i][j]:
                                        score_bbox = score_bbox+pred[k][4]
                                        cls_match_num = cls_match_num+1
                                elif (pred[k][5] == clue_txt_list[i][j] and pred[k][4] >= self.od_th):    # NOTE: 원하는 class (attribute 제외)가 th이상으로 detecting될 때
                                    score_bbox = score_bbox+pred[k][4]                                  # NOTE: bbox마다 score 계산
                                    cls_match_num = cls_match_num+1

                        if cls_match_num != 0:
                            score_txt = score_bbox / cls_match_num

                        if score_txt > self.txt_th:
                            od_detections = []
                            od_detector = apriltag.Detector()
                            od_detections.append(od_detector.detect(input_img))
                            od_tag_id = []

                            for j in range(0, len(od_detections[0])):
                                od_tag_id.append(od_detections[0][j].tag_id)
                            od_json_output = json_postprocess(clue_txts[i][-7:-5], od_tag_id)
                            self.json_list.append(od_json_output)
                        
                        if self.show_video:
                            for j in range(0, pred.shape[0]):
                                bboxes = pred[j][0:4]
                                confs = pred[j][4]
                                clss = pred[j][5]
                                upper_clss = pred[j][7]
                                lower_clss = pred[j][9]
                                ppl_clss = pred[j][11]

                                if clss == 0:   # NOTE: person
                                    if ppl_clss == 0:
                                        name = 'man'
                                    elif ppl_clss == 1:
                                        name = 'woman'
                                    else:
                                        name = 'child'

                                    upper_color = self.color_list[int(upper_clss.item())]
                                    lower_color = self.color_list[int(lower_clss.item())]

                                    label = f'{name} {float(confs):.2f} {upper_color} {lower_color}'
                                else:   # NOTE: object
                                    label = f'{self.names[int(clss)]} {float(confs):.2f}'

                                plot_one_box(bboxes, frame_for_vis, label=label, color=self.colors[int(clss)], line_thickness=2)
                            cv2.imwrite(self.debug_output_path+'frame'+str(self.cnt)+'_text_clue.jpg', frame_for_vis)

                        clue_info.append(clue_txts[i][-7:-5])

            # -----------------------------------------
            # Apriltag detection for room id
            # -----------------------------------------
            room_detections = []
            room_detector = apriltag.Detector()
            room_detections.append(room_detector.detect(input_img))
            room_tag_id = []
            for i in range(0, len(room_detections[0])):
                room_tag_id.append(room_detections[0][i].tag_id)

            for i in range(0, len(room_tag_id)):
                if room_tag_id[i] >= 500:
                    self.room_id = room_tag_id[i]

            # -----------------------------------------
            # json update and export
            # -----------------------------------------
            # NOTE: json dump에서 정답 json 만들기 (+중복 value 제거)
            for i in range(0, len(clue_info)):
                ans_list = []
                for j in range(0, len(self.json_list)):
                    k = list(self.json_list[j]['answer_sheet']['answer']['person_id'].keys())
                    for m in range(0, len(k)):
                        if clue_info[i] == k[m]:
                            v = self.json_list[j]['answer_sheet']['answer']['person_id'][k[m]]
                            for n in range(0, len(v)):
                                ans_list.append(v[n])
                self.json['answer_sheet']['answer']['person_id'][clue_info[i]] = list(set(ans_list))

            if self.room_id != None:                # NOTE: room id 저장
                self.json['answer_sheet']['room_id'] = str(self.room_id)

            self.cnt = self.cnt+1
            self.state = state

            ans_pair = self.json['answer_sheet']['answer']['person_id']
            ans_keys = list(ans_pair.keys())
            empty_cnt = 0
            for i in range(0, len(ans_keys)):
                if len(ans_pair[ans_keys[i]]) == 0:
                    empty_cnt = empty_cnt+1
                    self.json['answer_sheet']['answer']['person_id'][ans_keys[i]] = ["NONE"]
            if empty_cnt == len(clue_info):         # NOTE: value 전부 비어있으면 UNCLEAR로 채움
                for i in range(0, len(ans_keys)):
                    self.json['answer_sheet']['answer']['person_id'][ans_keys[i]] = ["UNCLEAR"]
            
            # print(self.cnt, self.state)
            return self.json

        except:
            self.json = {'answer_sheet': {
                        'room_id': None,
                        'mission': "1",
                        'answer': {
                            'person_id': 'UNCLEAR'
                            }   
                        }   
                    }
            # print('exception!')
            return self.json

if __name__ == "__main__":
    args = parse_args()
    task1 = Task1(args)

    if args.task1_debug == None:
        frames = None
    else:
        frames = args.debug_input_path              # NOTE: superglue 테스트이미지 (이미지 한장)

    task1(frames)
