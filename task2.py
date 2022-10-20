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


class Task2:
    def __init__(self,
                 yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
                 strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
                 config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 show_vid=False,  # show results
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 half=False,  # use FP16 half-precision inference
                 ):
        check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))

        # Load model
        self.device = select_device(device)
        self.half = half
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.cls_agnostic_nms = agnostic_nms
        self.show_video = show_vid

        WEIGHTS.mkdir(parents=True, exist_ok=True)
        self.model = attempt_load(Path(yolo_weights), map_location=self.device).eval()  # load FP32 model
        self.names, = self.model.names,
        self.stride = self.model.stride.max().cpu().numpy()  # model stride
        self.img_size = check_img_size(imgsz[0], s=self.stride)  # check image size

        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file(config_strongsort)

        # Create as many strong sort instances as there are video sources
        self.strong_sort = StrongSORT(
                    strong_sort_weights,
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

        # count dict
        self.count_dict = dict()
        for k in ['man', 'woman', 'child']:
            self.count_dict[k] = 0

    def __call__(self, original_img, augment=True):
        img = letterbox(original_img, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # Inference
        pred = self.model(img)

        # Apply NMS
        pred = non_max_suppression(pred[0], self.conf_thres, self.iou_thres, self.classes, self.cls_agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            self.curr_frames = original_img.copy()

            if self.strong_sort_ecc:  # camera motion compensation
                self.strong_sort.tracker.camera_update(self.prev_frames, self.curr_frames)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], original_img.shape).round()

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                outputs = self.strong_sort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), original_img)

                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output) in enumerate(outputs):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        c = int(cls)  # integer class
                        id = int(id)  # integer id
                        name = self.names[c]

                        if self.show_video:
                            label = f'{id} {self.names[c]} {conf:.2f}'
                            plot_one_box(bboxes, original_img, label=label, color=self.colors[c], line_thickness=2)

                        if name in self.target_list:
                            if (id, name) not in self.id_list:

                                self.id_list.append((id, name))

                                if name in self.man_list:
                                    self.count_dict['man'] += 1
                                if name in self.woman_list:
                                    self.count_dict['woman'] += 1
                                if name in self.child_list:
                                    self.count_dict['child'] += 1

            else:
                self.strong_sort.increment_ages()

            # Stream results
            if self.show_video:
                text = ''
                for k, v in self.count_dict.items():
                    text += f'{k}:{v} '
                cv2.putText(original_img, text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Visualize", original_img)
                cv2.waitKey(1)  # 1 millisecond

            self.prev_frames = self.curr_frames


if __name__ == "__main__":
    from tools.parse_args import parse_args
    args = parse_args()
    source = args.source
    del args.source

    task1 = Task2(**vars(args))
    dataset = LoadImages(source, img_size=args.imgsz, stride=task1.stride)

    with torch.no_grad():
        for _, _, original_frame, _ in dataset:
            task1(original_frame)
