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
from .utils.model_utils import matching, read_image, make_matching_plot
from .superglue.superpoint import SuperPoint
from .superglue.superglue import SuperGlue

from .yolov7.models.experimental import attempt_load
from .yolov7.utils.datasets import LoadImages
from .yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, cv2, xyxy2xywh
from .yolov7.utils.datasets import letterbox

CLASS_MAP = {
    'orange-skirt': 0, 'black-skirt': 1, 'blue-skirt': 2, 'box': 3, 'gray-skirt': 4, 'green-skirt': 5,
    'purple-skirt': 6, 'red-skirt': 7, 'trash-bin': 8, 'white-skirt': 9, 'yellow-skirt': 10, 'black_shirt': 11,
    'blue_shirt': 12, 'desk': 13, 'gray_shirt': 14, 'green_shirt': 15, 'orange_shirt': 16, 'purple_shirt': 17,
    'red_shirt': 18, 'white_shirt': 19, 'whiteboard': 20, 'yellow_shirt': 21, 'black_pants': 22, 'blue_pants': 23,
    'cabinet': 24, 'green_pants': 25, 'grey_pants': 26, 'monitor': 27, 'orange_pants': 28, 'purple_pants': 29,
    'red_pants': 30, 'white_pants': 31, 'yellow_pants': 32, 'basket': 33, 'bookshelf': 34, 'computer': 35,
    'laptop': 36, 'printer': 37, 'child': None, 'woman': None, 'man': None
}

class Task1:
    def __init__(self, args):
        self.clue_path = args.clue_path
        self.output_path = args.output_path
        self.task1_debug = args.task1_debug
        self.debug_path = args.debug_path
        self.img_conf_th = args.img_conf_th
        self.img_kp_th = args.img_kp_th
        self.txt_th = args.txt_th
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
        self.half = False
        self.conf_th = 0.25
        self.iou_th = 0.45
        self.classes = None
        self.cls_agnostic_nms = False
        self.yolo_path = args.yolo_path
        self.yolo = attempt_load(self.yolo_path, map_location='cuda').eval()
        self.stride = self.yolo.stride.max().cpu().numpy()
        self.img_size = check_img_size(self.imgsz[0], s=self.stride)
        self.names = self.yolo.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # -----------------------------------------
        # image clue preprocessing
        # -----------------------------------------
        clue_img_list = glob.glob(self.clue_path+'/*.png', recursive=True)   # png or jpg??
        clue_img_list.sort()
        self.clue_img_list = clue_img_list
        
        # -----------------------------------------
        # text clue preprocessing
        # -----------------------------------------
        txts = glob.glob(self.clue_path+'/*.json', recursive=True)
        txts.sort()
        txt_dict = json_preprocess(txts)
        
        txt_cls = []
        txt_dict_k = list(txt_dict.keys())
        for i in range(0, len(txt_dict_k)):
            txt_dict_v = txt_dict[txt_dict_k[i]]
            for j in range(0, len(txt_dict_v)):
                txt_cls.append(CLASS_MAP[txt_dict_v[j]])
        self.txt_clue = txt_cls

    def __call__(self, img):
        if self.task1_debug:
            input_img = img
        else:
            input_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_score = 0.0
        txt_score = 0.0

        # -----------------------------------------
        # Superglue
        # -----------------------------------------
        image0, inp0, scales0 = read_image(input_img, [640, 480], 'cuda', self.task1_debug)             # NOTE: video frame image
        
        clue_imgs = []
        clue_imgs_p = []
        clue_imgs_scales = []
        for clue_img_ in self.clue_img_list:
            if not self.task1_debug:
                clue_img_ = cv2.imread(clue_img_, cv2.IMREAD_GRAYSCALE)
            image1, inp1, scales1 = read_image(clue_img_, [640, 480], 'cuda', self.task1_debug)  # NOTE: clue image
            clue_imgs.append(image1)
            clue_imgs_p.append(inp1)
            clue_imgs_scales.append(scales1)

        if len(self.clue_img_list) > 0:
            score = []
            for i in range(len(self.clue_img_list)):    # NOTE: 각 이미지 단서마다 kpts, mean confidence 저장
                pred, matches, conf = matching({'image0': inp0, 'image1': clue_imgs_p[i]}, self.superpoint, self.superglue)
                kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]    # NOTE: superpoint 개수
                score.append((mkpts0.shape[0], mconf.mean()))

                # for debugging
                if self.debug_path != None:
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
                                        self.debug_path+'_frame'+str(self.cnt)+'_clue'+str(i), small_text)
                
            img_score = 0.0
            for i in range(0, len(score)):
                if score[i][0] >= self.img_kp_th:
                    img_score = img_score+score[i][1]

        # -----------------------------------------
        # YOLO
        # -----------------------------------------
        if len(self.txt_clue) > 0:
            # load_img = LoadImages(img, img_size=self.imgsz, stride=self.stride)
            # original_img = next(iter(load_img))[2]
            original_img = img
            yolo_img = letterbox(original_img, self.img_size, stride=self.stride)[0]
            yolo_img = yolo_img[:, :, ::-1].transpose(2, 0, 1)
            yolo_img = np.ascontiguousarray(yolo_img)
            yolo_img = torch.from_numpy(yolo_img).to('cuda')
            yolo_img = yolo_img.half() if self.half else yolo_img.float()
            if len(yolo_img.shape) == 3:
                yolo_img = yolo_img[None]

            obj_detections = self.yolo(yolo_img)
            obj_detections = non_max_suppression(obj_detections[0], self.conf_th, self.iou_th, self.classes, self.cls_agnostic_nms) # TODO: 클래스추가

            det = obj_detections[0]
            det[:, :4] = scale_coords(yolo_img.shape[2:], det[:, :4], original_img.shape).round()
            xywhs = xyxy2xywh(det[:, 0:4])  # bbox
            confs = det[:, 4]   # conf
            clss = det[:, 5]    # class

            # score 계산
            for i in range(0, len(self.txt_clue)):
                for j in range(0, len(det)):
                    if self.txt_clue[i] == clss[j] and confs[j] >= 0.6:
                        txt_score = txt_score+confs[j]
            txt_score = txt_score / len(self.txt_clue)

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

        # room id here  TODO: room id 고정
        room_id = None
        for i in range(0, len(tag_id)):
            if tag_id[i] >= 500:
                room_id = tag_id[i]

        # -----------------------------------------
        # json export
        # -----------------------------------------
        total_score = img_score+txt_score
        if len(self.clue_img_list) > 0 and len(self.txt_clue) == 0:  # image clue only
            if img_score > self.img_conf_th:
                json_output = json_postprocess(clues_num, data, room_id)
            else:
                json_output = None
        elif len(self.clue_img_list) == 0 and len(self.txt_clue) > 0: # txt clue only
            if txt_score > self.txt_th:
                json_output = json_postprocess(clues_num, data, room_id)
            else:
                json_output = None
        elif len(self.clue_img_list) > 0 and len(self.txt_clue) > 0: # image and txt clue
            if total_score > self.total_th:
                json_output = json_postprocess(clues_num, data, room_id)
            else:
                json_output = None
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, indent=4)

        print(json.dumps(json_output, ensure_ascii=False, indent=4))

        self.cnt = self.cnt+1

if __name__ == "__main__":
    args = parse_args()
    task1 = Task1(args)

    if args.task1_debug == None:
        frames = None
    else:
        frames = '/home/eulrang/workspace/git/Drone_Challenge/task1/toy_test/image_matching/image0.jpg' # NOTE: superglue 테스트이미지 (이미지 한장)

    task1(frames)