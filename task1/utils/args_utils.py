import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default=None, help='video path')
    parser.add_argument('--clue_path', default=None, help='clue(img, txt) path')        # TODO: img -> hint 수정함 # 경로지정 필수
    parser.add_argument('--output_path', default='.', help='output path')
    parser.add_argument('--debug_path', default='.', help='debugging output path')
    parser.add_argument('--frame_skip', type=int, default=30, help='output path')
    parser.add_argument('--img_conf_th', type=int, default=0.2, help='img threshold')   # TODO: determine best confidence threshold value
    parser.add_argument('--img_kp_th', type=int, default=5, help='img threshold')       # TODO: determine best keypoint threshold value
    parser.add_argument('--txt_th', type=int, default=0.8, help='txt threshold')        # for Yolo FIXME
    args = parser.parse_args()

    return args