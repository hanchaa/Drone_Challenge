import numpy as np
import torch
import cv2
from .utils.model_utils import match_pairs#, detect_objects
from .utils.common_utils import save_match_imgs

def img_model(frames, img_clue, img_save_path, conf_th, kp_th, search_radius=15, match_batch_size=1):    # TODO: search_radius range fix??
    if not img_clue:
        return 0
    else:
        # frame_total = len(frames)
        frame_total = 0
        print('start image matching')
        # img = cv2.imread(frames, cv2.IMREAD_GRAYSCALE)
        match_results, all_score = match_pairs(frames, img_clue, match_batch_size, 'cpu')
        print("match_results:", match_results)
        torch.cuda.empty_cache()

        # mask frames 
        # TODO: multi-clue
        vid_mask = np.zeros(frame_total).astype(np.int32)
        img_idx = []
        for match_res in match_results:
            img_idx.append(match_res[0]) 
            if match_res[0] == -1: 
                continue
            idx = np.arange(search_radius*2+1) - search_radius + match_res[0]
            idx = np.clip(idx,0,frame_total-1).astype(np.int32)
            vid_mask[idx] = 1
        masked_frame_idx = np.where(vid_mask==1)[0] # 매칭되는 frame index
        matched_frames = np.stack(frames, axis=0)[vid_mask==1] # 매칭되는 frame image들만 뽑은 리스트 
        # matched_score = all_score[0][masked_frame_idx[0]:masked_frame_idx[-1]+1] # TODO: make it pretty for multi-clue (e.g., matched_score = all_score[N][:])
        matched_score = all_score
        # for debugging
        # save_match_imgs(masked_frame_idx, matched_frames, img_save_path)

        result = []
        for i in range(0, len(matched_score)):
            if len(matched_score[i]) >= kp_th:
                score = matched_score[i].sum()/len(matched_score[i])    
                if score >= conf_th:
                    result.append([matched_frames[i], score])  # TODO: make it pretty (e.g., result_im[0] = [frames[0], scores[0]])

        return result

def txt_model(frames, txts, img_save_path, box_th):
    if not txts:
        return 0
    else:
        print('now working')