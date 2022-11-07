import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages, LoadStreams
from yolov7.utils.general import (check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, check_file)
from yolov7.utils.torch_utils import select_device, time_synchronized
from yolov7.utils.plots import plot_one_box
from yolov7.utils.datasets import letterbox
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


class Task2Vision:
    def __init__(self, args):
        
        # Load model
        self.device = select_device(args.device)
        self.half = args.half
        self.conf_thres = args.conf_thres
        self.iou_thres = args.iou_thres
        self.classes = args.classes
        # self.classes = []
        self.cls_agnostic_nms = args.agnostic_nms
        self.show_video = args.show_vid

        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.model = attempt_load(Path(args.yolo_weights), map_location=self.device).eval()  # load FP32 model
        self.names, = self.model.names,
        self.stride = self.model.stride.max().cpu().numpy()  # model stride
        self.img_size = check_img_size(args.imgsz[0], s=self.stride)  # check image size

        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file(args.config_strongsort)

        # Create as many strong sort instances as there are video sources
        self.strong_sort = StrongSORT(
                    args.strong_sort_weights,
                    self.device,
                    self.half,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,
                )
        self.strong_sort.model.warmup()
        self.strong_sort_ecc = cfg.STRONGSORT.ECC

        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        # Run tracking
        self.curr_frames, self.prev_frames = None, None

        self.id_list = []
        self.target_list = ['black_man', 'man', 'lying_man', 'woman', 'lying_woman', 'lying_child', 'child']
        self.man_list = ['black_man', 'man', 'lying_man']
        self.woman_list = ['woman', 'lying_woman']
        self.child_list = ['child', 'lying_child']
        self.delete_list = ['lifeguard', 'medical staff', 'poster_image']
        self.color_list = ['OCC','RED','ORG','YLW','GRN','BLU','PRP','WHT','GRY','BLK']

        # count dict
        self.count_dict = dict()
        for k in ['man', 'woman', 'child']:
            self.count_dict[k] = 0
        self.prev_room_return_sheet = None
        self.prev_state = -1
        self.UNCLEAR_THRES = args.unclear_thres

    def __call__(self, original_img, state, frame_for_vis=None):
        img = letterbox(original_img, self.img_size, stride=self.stride)[0]
        FRAME_DATA_PARSE = dict()

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred_all = self.model(img)
        pred_loc_obj = pred_all[0][:,:,:27]
        pred_uppercol = pred_all[0][:,:,27:37]
        pred_lowercol = pred_all[0][:,:,37:47]
        pred_ppltype = pred_all[0][:,:,47:50]

        # Apply NMS
        pred = non_max_suppression(pred_all[0], self.conf_thres, self.iou_thres, self.classes, self.cls_agnostic_nms, multi_label=False, return_attributes=True)        
        # pred[0] : [X, Y, W, H, cls_conf, cls, upper_conf, upper_cls, lower_conf, lower_cls, ppl_conf, ppl_cls, oth_conf, oth_cls]
        # @ TASK1 WORKERS : detection output of all objects, along with attribute confidences and classes
        FRAME_DATA_PARSE['object_bbox'] = pred[0][:, :4]
        FRAME_DATA_PARSE['object_class'] = pred[0][:, 5:6]
        FRAME_DATA_PARSE['object_confidence'] = pred[0][:, 4:5]
        FRAME_DATA_PARSE['object_is_poster'] = pred[0][:, 5:6] == 18
        FRAME_DATA_PARSE['object_is_person'] = pred[0][:, 5:6] == 0
        FRAME_DATA_PARSE['upper_color_confidence'] = pred[0][:, 6:7]
        FRAME_DATA_PARSE['upper_color_class'] = pred[0][:, 7:8]
        FRAME_DATA_PARSE['lower_color_confidence'] = pred[0][:, 8:9]
        FRAME_DATA_PARSE['lower_color_class'] = pred[0][:, 9:10]
        FRAME_DATA_PARSE['person_type_confidence'] = pred[0][:, 10:11]
        FRAME_DATA_PARSE['person_type_class'] = pred[0][:, 11:12]

        
        # Process detections

        # remove poster person
        person_pred = pred[0][pred[0][:, 5] == 0]
        not_person_pred = pred[0][pred[0][:, 5] != 0]
        poster_pred = pred[0][pred[0][:, 5] == 18]
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
            pred = [torch.cat([person_pred, not_person_pred])]

        for i, det in enumerate(pred):  # detections per image
            self.curr_frames = original_img.copy()

            if self.strong_sort_ecc:  # camera motion compensation
                self.strong_sort.tracker.camera_update(self.prev_frames, self.curr_frames)

            # @ TASK1 WORKERS : this considers only person class!
            det = det[det[:,5] == 0]
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_img.shape).round()

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                upper_confs = det[:, 6]
                upper_clss = det[:, 7]
                lower_confs = det[:, 8]
                lower_clss = det[:, 9]
                ppl_confs = det[:, 10]
                ppl_clss = det[:, 11]
                outputs = self.strong_sort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), original_img,
                                                  attributes=[upper_clss.cpu(),lower_clss.cpu(),ppl_clss.cpu()])
                # @ TASK1 WORKERS : tracking output of 'person' class only; need to modify the line above.

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output) in enumerate(outputs):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        upper_cls = output[7]
                        lower_cls = output[8]
                        ppl_cls = output[9]
                        if ppl_cls == 0 :
                            name = 'man'
                        elif ppl_cls == 1 :
                            name = 'woman'
                        else :
                            name = 'child'
                        upper_color = self.color_list[int(upper_cls.item())]
                        lower_color = self.color_list[int(lower_cls.item())]
                        # c = int(cls)  # integer class
                        id = int(id)  # integer id
                        # name = self.names[c]
                        
                        if self.show_video:
                            # label = f'{id} {self.names[c]} {conf:.2f}'
                            label = f'{id} {name} {conf:.2f} {upper_color} {lower_color}'
                            plot_one_box(bboxes, original_img, label=label, color=self.colors[c], line_thickness=2)

                        if (id, name) not in self.id_list :
                            self.id_list.append((id, name))

                            if name == 'man':
                                self.count_dict['man'] += 1
                            if name == 'woman':
                                self.count_dict['woman'] += 1
                            if name == 'child':
                                self.count_dict['child'] += 1


                        # if name in self.target_list and state % 2 == 1:
                        #     if (id, name) not in self.id_list:

                        #         self.id_list.append((id, name))

                        #         if name in self.man_list:
                        #             self.count_dict['man'] += 1
                        #         if name in self.woman_list:
                        #             self.count_dict['woman'] += 1
                        #         if name in self.child_list:
                        #             self.count_dict['child'] += 1

            else:
                self.strong_sort.increment_ages()

            
            ANSWER_PARSE = self.answer_parser(state)

            # Stream results
            if self.show_video:
                text = ''
                for k, v in self.count_dict.items():
                    text += f'{k}:{v} '
                cv2.putText(original_img, text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Visualize", original_img)
                cv2.waitKey(1)  # 1 millisecond

            self.prev_frames = self.curr_frames

        return ANSWER_PARSE, FRAME_DATA_PARSE


    def answer_parser(self,state):
        # TODO SANITY CHECK!!!!! 
        # init if needed
        if self.prev_state == 0 and state > 0 :  # = just entered room
            for k in ['man', 'woman', 'child']:
                self.count_dict[k] = 0

        # parse data
        if state > 0 :
            room_id = str(state)
            self.prev_room = state

            return_sheet = dict()
            answer_sheet = dict()
            answer_sheet["room_id"] = room_id 
            answer_sheet["mission"] = "2"
            count_format = dict()
            m = str(self.count_dict['man']) if self.count_dict['man'] < self.UNCLEAR_THRES else 'UNCLEAR'
            w = str(self.count_dict['woman']) if self.count_dict['woman'] < self.UNCLEAR_THRES else 'UNCLEAR'
            c = str(self.count_dict['child']) if self.count_dict['child'] < self.UNCLEAR_THRES else 'UNCLEAR'
            count_format["person_num"] = {"M": m, "W": w, "C": c}
            answer_sheet["answer"] = count_format
            return_sheet['answer_sheet'] = answer_sheet

            self.prev_room_return_sheet = return_sheet
            self.prev_state = state

            return return_sheet

        elif state == 0 :
            self.prev_state = state
            if self.prev_room_return_sheet is not None :
                return self.prev_room_return_sheet
            else :
                room_id = "-1"
                return_sheet = dict()
                answer_sheet = dict()
                answer_sheet["room_id"] = room_id 
                answer_sheet["mission"] = "2"
                count_format = dict()
                count_format["person_num"] = {"M": '0', "W": '0', "C": '0'}
                answer_sheet["answer"] = count_format
                return_sheet['answer_sheet'] = answer_sheet

                self.prev_room_return_sheet = return_sheet
                return return_sheet
                
        elif state == -1 :
            room_id = "-1"
            self.prev_room = -1
            self.prev_state = state

            return_sheet = dict()
            answer_sheet = dict()
            answer_sheet["room_id"] = room_id 
            answer_sheet["mission"] = "2"
            count_format = dict()
            count_format["person_num"] = {"M": '0', "W": '0', "C": '0'}
            answer_sheet["answer"] = count_format
            return_sheet['answer_sheet'] = answer_sheet

            self.prev_room_return_sheet = return_sheet
            
            return return_sheet

        # if state == 0:
        #     room_id = state
        # else:
        #     if state % 2 == 0:
        #         room_id = state - 1
        #     else:
        #         room_id = state
        # self.prev_state = state
        # return return_sheet




# if __name__ == "__main__":
#     from tools.parse_args import parse_args
#     args = parse_args()
#     source = "task2_vision/yolov7/video/set03_drone03.mp4"
#     task2vision = Task2Vision(args)
#     # task2vision = Task2Vision(**vars(args))
#     dataset = LoadImages(source, img_size=args.imgsz, stride=task1.stride)
#     import pdb;pdb.set_trace()
#     with torch.no_grad():
#         for _, _, original_frame, _ in dataset:
#             task2vision(original_frame)
