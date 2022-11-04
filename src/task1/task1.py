import cv2
import glob
import json
import apriltag
import numpy as np
import matplotlib.cm as cm
from numpy import random
import torch

from .utils.json_utils import json_preprocess, json_postprocess
from .utils.args_utils import parse_args
from .utils.image_matching_utils import matching, read_image, make_matching_plot
from .utils.od_utils import plot_one_box
from .superglue.superpoint import SuperPoint
from .superglue.superglue import SuperGlue

from .yolov7.models.experimental import attempt_load
from .yolov7.utils.datasets import letterbox, LoadImages
from .yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, cv2
from .yolov7.utils.torch_utils import TracedModel


CLASS_MAP = {
    'orange skirt': 0, 'black skirt': 1, 'blue skirt': 2, 'box': 3, 'gray skirt': 4, 'green skirt': 5,
    'purple skirt': 6, 'red skirt': 7, 'trash-bin': 8, 'white skirt': 9, 'yellow skirt': 10, 'black shirt': 11,
    'blue shirt': 12, 'desk': 13, 'gray shirt': 14, 'green shirt': 15, 'orange shirt': 16, 'purple shirt': 17,
    'red shirt': 18, 'white shirt': 19, 'whiteboard': 20, 'yellow shirt': 21, 'black pants': 22, 'blue pants': 23,
    'cabinet': 24, 'green pants': 25, 'gray pants': 26, 'monitor': 27, 'orange pants': 28, 'purple pants': 29,
    'red pants': 30, 'white pants': 31, 'yellow pants': 32, 'basket': 33, 'bookshelf': 34, 'computer': 35,
    'laptop': 36, 'printer': 37, 'child': None, 'woman': None, 'man': None
}

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
        self.cnt = 0

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
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        if self.half:
            self.yolo = TracedModel(yolo, 'cuda', self.img_size).half()
        else:
            self.yolo = TracedModel(yolo, 'cuda', self.img_size)

    def __call__(self, img: np.ndarray, state):
        try:
            img_score = 0.0
            txt_score = 0.0
            clue_txt = []
            
            # -----------------------------------------
            # text clue preprocessing
            # -----------------------------------------
            txts = glob.glob(self.clue_path+'/*.json', recursive=True)
            if len(txts) > 0:
                txts.sort()
                txt_dict = json_preprocess(txts)
            
                txt_cls = []
                txt_dict_k = list(txt_dict.keys())
                for i in range(0, len(txt_dict_k)):
                    txt_dict_v = txt_dict[txt_dict_k[i]]
                    for j in range(0, len(txt_dict_v)):
                        txt_cls.append(CLASS_MAP[txt_dict_v[j]])
                txt_clue = txt_cls

            # -----------------------------------------
            # image clue preprocessing
            # -----------------------------------------
            clue_img_list = glob.glob(self.clue_path+'/*.png', recursive=True)   # png or jpg??
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
                input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            image0, inp0, scales0 = read_image(input_img, [640, 480], 'cuda')       # NOTE: video frame image


            if len(clue_img_list) > 0:
                score = []
                for i in range(len(clue_img_list)):    # NOTE: 각 이미지 단서마다 kpts, mean confidence 저장
                    pred, matches, conf = matching({'image0': inp0, 'image1': clue_imgs_p[i]}, self.superpoint, self.superglue)
                    kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                    valid = matches > -1
                    mkpts0 = kpts0[valid]
                    mkpts1 = kpts1[matches[valid]]
                    mconf = conf[valid]    # NOTE: superpoint 개수
                    score.append((mkpts0.shape[0], mconf.mean()))

                    # for debugging
                    if self.debug_output_path != None:
                        if (score[i][0] > 50 and score[i][1] > 0.7):    # superpoint > 50 & confidence > 0.5 일 때만 이미지 저장
                            color = cm.jet(mconf)
                            label = [
                                'SuperGlue',
                                'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
                                'Matches: {}'.format(len(mkpts0)),
                            ]

                            k_thresh = self.img_config['superpoint']['keypoint_threshold']
                            m_thresh = self.img_config['superglue']['match_threshold']
                            small_text = [
                                'Keypoint Threshold: {:.4f}'.format(k_thresh),
                                'Match Threshold: {:.2f}'.format(m_thresh),
                            ]

                            make_matching_plot(image0, clue_imgs[i], mkpts0, mkpts1, color, label,
                                            self.debug_output_path+'frame'+str(self.cnt)+'_clue'+str(i), small_text)

                    clue_txt.append(clue_img_list[i][-6:-4])

                img_score = 0.0
                for i in range(0, len(score)):
                    if score[i][0] >= self.img_kp_th:
                        img_score = img_score+score[i][1]


            # -----------------------------------------
            # YOLO inference
            # -----------------------------------------
            if len(txt_clue) > 0:
                self.yolo(torch.zeros(1, 3, self.img_size, self.img_size).to('cuda').type_as(next(self.yolo.parameters())))
                if self.task1_debug:
                    load_img = LoadImages(img, img_size=self.imgsz, stride=self.stride)
                    _, yolo_img, im0s, _ = next(iter(load_img))
                else:
                    im0s = img
                    yolo_img = letterbox(im0s, self.img_size, stride=self.stride)[0]
                    yolo_img = yolo_img[:, :, ::-1].transpose(2, 0, 1)
                    yolo_img = np.ascontiguousarray(yolo_img)
                yolo_img = torch.from_numpy(yolo_img).to('cuda')
                yolo_img = yolo_img.half() if self.half else yolo_img.float()
                yolo_img /= 255.0
                if len(yolo_img.shape) == 3:
                    yolo_img = yolo_img.unsqueeze(0)

                old_img_w = old_img_h = self.img_size
                old_img_b = 1

                # Warmup
                if (old_img_b != yolo_img.shape[0] or old_img_h != yolo_img.shape[2] or old_img_w != yolo_img.shape[3]):
                    old_img_b = yolo_img.shape[0]
                    old_img_h = yolo_img.shape[2]
                    old_img_w = yolo_img.shape[3]
                    for i in range(3):
                        self.yolo(yolo_img)[0]

                with torch.no_grad():
                    pred = self.yolo(yolo_img)[0]
                pred = non_max_suppression(pred, self.conf_th, self.iou_th, self.classes, self.cls_agnostic_nms)[0]
                pred[:, :4] = scale_coords(yolo_img.shape[2:], pred[:, :4], im0s.shape).round()

                if (self.task1_debug or self.debug_output_path != None):
                    for *xyxy, conf, cls in reversed(pred):
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0s, label=label, color=self.colors[int(cls)], line_thickness=1)
                    cv2.imwrite(self.debug_output_path+'frame'+str(self.cnt)+'_text_clue.jpg', im0s)

                for i in range(0, len(txts)):                
                    clue_txt.append(txts[i][-7:-5])

                # score 계산
                for i in range(0, len(txt_clue)):
                    for j in range(0, pred.shape[0]):
                        if (txt_clue[i] == pred[j][5] and pred[j][4] >= self.od_th):     # NOTE: prediction = [x, y, x, y, conf, cls]
                            txt_score = txt_score+pred[j][4]

            # -----------------------------------------
            # Apriltag detection
            # -----------------------------------------
            april_detections = []
            tag_id = []
            detector = apriltag.Detector()
            april_detections.append(detector.detect(input_img))
            tag_id = []
            for i in range(0, len(april_detections[0])):
                tag_id.append(april_detections[0][i].tag_id)

            tag_id_set = set(tag_id)
            tag_id_list = list(tag_id_set)
            img_clue_num = len(april_detections)
            clues_num = img_clue_num # txt_clue_num
            data = [clues_num, tag_id_list]

            room_id = None
            for i in range(0, len(tag_id)):
                if tag_id[i] >= 500:
                    room_id = tag_id[i]

            # -----------------------------------------
            # json export
            # -----------------------------------------
            total_score = img_score+txt_score
            json_output = json_postprocess(clue_txt, data, room_id, unclear=True)   # json UNCLEAR initialization
            if len(clue_img_list) > 0 and len(txt_clue) == 0:  # image clue only
                if img_score > self.img_conf_th:
                    json_output = json_postprocess(clue_txt, data, room_id)
            elif len(clue_img_list) == 0 and len(txt_clue) > 0: # txt clue only
                if txt_score > self.txt_th:
                    json_output = json_postprocess(clue_txt, data, room_id)
            elif len(clue_img_list) > 0 and len(txt_clue) > 0: # image and txt clue
                if total_score > self.total_th:
                    json_output = json_postprocess(clue_txt, data, room_id)
            
            # with open(self.json_output_path, 'w', encoding='utf-8') as f:
            #     json.dump(json_output, f, indent=4)

            print(room_id, json_output)

            self.cnt = self.cnt+1

            return json_output

        except:
            pass

if __name__ == "__main__":
    args = parse_args()
    task1 = Task1(args)

    if args.task1_debug == None:
        frames = None
    else:
        frames = args.debug_input_path # NOTE: superglue 테스트이미지 (이미지 한장)

    task1(frames)