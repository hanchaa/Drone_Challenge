import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', type=str, help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str)
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    return args
