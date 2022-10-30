import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # task 1
    parser.add_argument('--clue_path', default="/home/eulrang/workspace/git/AGC2022_round3_task1/data_toy/drone_task1_toy/case3", help='clue(img, txt) path')
    parser.add_argument('--output_path', default='/home/eulrang/workspace/git/AGC2022_round3_task1/data_toy/outputs/result.json', help='output path')
    parser.add_argument('--task1_debug', action="store_true", help='(optional)debug mode')
    parser.add_argument('--debug_path', default=None, help='debugging output path')
    parser.add_argument('--yolo_path', default='/home/eulrang/workspace/git/Drone_Challenge/task1/yolov7/models/best.pt', help='yolo task1 checkpoint path')
    parser.add_argument('--img_conf_th', type=int, default=0.2,
                        help='img threshold')  # TODO: determine best confidence threshold value
    parser.add_argument('--img_kp_th', type=int, default=5,
                        help='img threshold')  # TODO: determine best keypoint threshold value
    parser.add_argument('--txt_th', type=int, default=0.8, help='txt threshold')        # TODO: determine best value
    parser.add_argument('--total_th', type=int, default=0.9, help='img+txt threshold')  # TODO: determine best value

    # task 2
    parser.add_argument('--yolo-weights', type=str, help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str)
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=(640, 640), help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', default=False, help='display tracking video results')
    parser.add_argument('--classes', type=int, default=None, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', default=False, help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', default=False, help='use FP16 half-precision inference')

    # for debugging
    parser.add_argument('--video_path', type=str, default=None, help='video path')

    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    return args
