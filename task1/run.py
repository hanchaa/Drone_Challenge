import time
import cv2
import argparse
import glob
import json
import apriltag
from utils.common_utils import MatchImageSizeTo, save_tag_match_imgs
from utils.json_utils import json_postprocess
from model import img_model, txt_model

if __name__ == '__main__':
    start = time.time()

    # -----------------------------------------
    # args parsing
    # -----------------------------------------
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

    # -----------------------------------------
    # image clue preprocessing
    # TODO: make it pretty
    # -----------------------------------------
    img_list = glob.glob(args.clue_path+'/*.png', recursive=True)
    img_list.sort()
    imgs=[]
    img_resizer = MatchImageSizeTo()
    for img_ in img_list:
        img = cv2.imread(img_, cv2.IMREAD_GRAYSCALE)
        img = img_resizer(img)
        imgs.append(img)

    # # -----------------------------------------
    # # text clue preprocessing
    # # -----------------------------------------
    # txt_list = glob.glob(args.clue_path+'/*.json', recursive=True)
    # txt_list.sort()
    
    # -----------------------------------------
    # video input preprocessing 
    # TODO: real-time video input handling
    # -----------------------------------------
    frames = []
    cap = cv2.VideoCapture(args.video_path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        frame_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if(type(frame) == type(None)):
            break
        if frame_pos % args.frame_skip != 0:
            continue
        frames.append(frame)
    cap.release()
    
    # -----------------------------------------
    # model inference
    # -----------------------------------------
    result_im = img_model(frames, imgs, args.debug_path, args.img_conf_th, args.img_kp_th)  # [[img, score], [img, score], ..., [img, score]]
    # result_txt = txt_model(frames, txt_list, args.debug_path, args.txt_th)    # TODO: add text condition

    # -----------------------------------------
    # apriltag detection
    # -----------------------------------------
    # for debugging
    # save_tag_match_imgs(image, detections, args.debug_path)
    images = []
    detections = []
    for i in range(0, len(result_im)):
        images.append(cv2.cvtColor(result_im[i][0], cv2.COLOR_BGR2GRAY))
        detector = apriltag.Detector()
        detections.append(detector.detect(images[i]))
        tag_id = []
        if len(detections[i]) == 0:
            tag_id = []
        else:
            for j in range(0, len(detections)):
                if len(detections[j]) == 0:
                    continue
                else:
                    for k in range(0, len(detections[j])):
                        tag_id.append(detections[j][k].tag_id)
        

    # -----------------------------------------
    tag_id_set = set(tag_id)                # TODO: tag_id - tag_name pair
    tag_id_list = list(tag_id_set)
    clue_num = [1]                          # consider case 1   # TODO: add clue_num module
    data = [clue_num[0], tag_id_list]       # TODO: generalize clue_num
    # -----------------------------------------

    # -----------------------------------------
    # json export
    # num_clues = len(imgs)   # TODO: txt 추가
    num_clues = len(clue_num)
    json_output = json_postprocess(num_clues, data)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=4)
    # -----------------------------------------
    
    print(json.dumps(json_output, ensure_ascii=False, indent=4))
    print("TIME :", time.time()-start)