import numpy as np
import torch
import math
import cv2
from tqdm import tqdm
from ..superglue.superpoint import SuperPoint
from ..superglue.superglue import SuperGlue

# -----------------------------------------
# Superglue
# -----------------------------------------
def match_pairs(vid_, imgs, vid_batch, device,
                match_num_rate_threshold=0.01,
                superglue='indoor', 
                max_keypoints = 1024, 
                keypoint_threshold = 0.0, 
                nms_radius = 4, 
                sinkhorn_iterations = 20, 
                match_threshold = 0.2):
    """ 
    Args: 
        vid_: list of numpy vid frames, range 0~255, shape H x W x 3 , BGR           
        imgs: list of numpy images, range 0~255, shape H x W , Grayscale
        vid_batch: batch size for video
        device: device
    Return:
        result: list of tuples (frame idx, match_rate)
    """

    # vid = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in vid_]
    
    torch.set_grad_enabled(False)

    config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': superglue,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

    superpoint = SuperPoint(config.get('superpoint', {})).eval().to(device)
    superglue = SuperGlue(config.get('superglue', {})).eval().to(device)

    # T = len(vid)
    # N = len(imgs)
    T = 1
    N = 1
    
    imgs = [torch.from_numpy(imgs[i]/255.).float()[None,None] for i in range(N)]  # (1,1,H,W)
    imgs_kp = []
    match_num_threshold = []

    # image clues
    for img in imgs:
        img = img.to(device)
        kp = superpoint({'image': img})    # 'keypoints', 'scores', 'descriptors'
        kp = {**{k+'0': v for k, v in kp.items()}}
        for k in kp:
            if isinstance(kp[k], (list,tuple)):
                kp[k] = torch.stack(kp[k])    # (1,K,2), (1,K), (1,D,K)
        imgs_kp.append(kp)
        match_num_threshold.append(int(kp['keypoints0'].shape[1]*match_num_rate_threshold))



    vid = vid_
    result = [[-1,0] for i in range(N)]
    score = [[] for i in range(N)]
    # vid_size = vid[0].shape[-2:]
    vid_size = vid.shape
    Iters = math.ceil(T/vid_batch)
    start = 0

    with tqdm(total=Iters) as pbar:
        for i in range(Iters):
            start = i * vid_batch
            if i == Iters-1:
                end = T
            else:
                end = (i+1) * vid_batch
            # frames = [torch.from_numpy(vid[i]/255.).float()[None] for i in range(start,end)] #(1,H,W)
            # frames = torch.stack(frames).to(device)  # (B,1,H,W)
            frames = torch.from_numpy(vid).float().unsqueeze(0).unsqueeze(0)
            vid_kp = superpoint({'image':frames})
            vid_kp = {**{k+'1': v for k, v in vid_kp.items()}}
            for k in vid_kp:
                if isinstance(vid_kp[k], (list,tuple)):
                    vid_kp[k] = torch.stack(vid_kp[k])    # (B,K,2), (B,K), (B,D,K)

            for n, img_kp_ in enumerate(imgs_kp):
                img_size = imgs[n].shape[-2:]
                img_kp = {}
                for k in img_kp_:
                    if len(img_kp_[k].shape)==2:
                        img_kp[k] = img_kp_[k].repeat((end-start),1) # (B,K,2), (B,K), (B,D,K)
                    else:
                        img_kp[k] = img_kp_[k].repeat((end-start),1,1) # (B,K,2), (B,K), (B,D,K)

                data = {**vid_kp, **img_kp, 'image0_shape': img_size, 'image1_shape': vid_size}
                pred = superglue(data)  # matches0, matches1, matching_scores0, matching_scores1
                pred = {k:v.cpu().numpy() for k,v in pred.items()} # all (B,~1024)
                valid = pred['matches0']>-1
                match_scores = pred['matching_scores0'] # TODO: double check
                match_num = np.sum(valid, axis=1) # (B,)
                max_idx = np.argmax(match_num)

                for i in range(0, end-start):
                    # for debugging
                    # match_conf = match_scores[i][valid[i]]
                    # print('idx[{}]: confidence={}, match keypoints={}'.format(i+start, match_conf, len(match_conf)))

                    score[n].append(match_scores[i][valid[i]])

                if match_num[max_idx] < match_num_threshold[n]:
                    continue
                elif match_num[max_idx] < result[n][1]:
                    continue
                else:
                    result[n][1] = match_num[max_idx]
                    result[n][0] = start + max_idx
            pbar.update(1)
    
    # return result, score
    return result, match_scores


# -----------------------------------------
# YOLOv7
# -----------------------------------------
# def detect_objects(vid_, imgs, vid_batch, device,
#                     ...):
#     # Define Yolov7 model




#     return result