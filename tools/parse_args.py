import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # task 1
    parser.add_argument('--clue-path', default=None, help='clue(img, txt) path')  # TODO: img -> hint 수정함 # 경로지정 필수
    parser.add_argument('--output-path', default='.', help='output path')
    parser.add_argument('--debug-path', default='.', help='debugging output path')
    parser.add_argument('--frame-skip', type=int, default=30, help='output path')
    parser.add_argument('--img-conf-th', type=int, default=0.2,
                        help='img threshold')  # TODO: determine best confidence threshold value
    parser.add_argument('--img-kp-th', type=int, default=5,
                        help='img threshold')  # TODO: determine best keypoint threshold value
    parser.add_argument('--txt-th', type=int, default=0.8, help='txt threshold')  # for Yolo FIXME
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
    parser.add_argument('--video-path', type=str, default=None, help='video path')

    args = parser.parse_args()
    args.imgsz *= 2 if len(args.imgsz) == 1 else 1  # expand

    return args
