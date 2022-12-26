import torch
import librosa
import numpy as np
import julius
import glob
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display
import os

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

def consecutive_merge(data, threshold=10):
    merge = []
    if len(data) == 1 and len(data[0])==0:
        return np.asarray(merge)
    else:
        temp = [data[0][0], data[0][-1]]
        for interval in data[1:]:
            if temp[-1] + threshold > interval[0]:
                temp[1] = interval[-1]
            else:
                merge.append(temp)
                temp = [interval[0], interval[-1]]
        merge.append(temp)
        return np.asarray(merge)

def seg_to_answer(segment, prob, min_frame=3):
    answer = []
    frame = np.zeros_like(prob)
    for item in segment:
        start, end = item
        if end-start > min_frame:
            weighted_center = (round)( (prob[start:end] * np.arange(start, end)).sum() / prob[start:end].sum() * (1600/16000) )
            answer.append(weighted_center)
            # mid_point = (round)(np.mean(item)/10)
            frame[start:end] = 2
            frame[(int)(weighted_center*16000/1600)-3:(int)(weighted_center*16000/1600)+3] = 1
    if len(answer) == 0:
        answer = [None]
    return np.asarray(answer), frame

def convert_intsec_to_strsec(sec):
    if sec == None:
        strsec = None
    else:
        min_str = str(sec // 60)
        sec_str = str(sec % 60)
        strsec = min_str.zfill(2) + ':' + sec_str.zfill(2)
    return strsec

def cw_metrics_cal(gt, pred):
    correct = 0
    deletion = 0
    insertion = 0
    include_list = []
    for ii in range(len(gt)):
        is_include = 0
        if gt[ii][0] == None:
            if pred[0] == None:
                correct += 1
        else:
            if pred[0] == None:
                deletion += 1
            else:
                gt_start, gt_end = gt[ii][0], gt[ii][1]
                for jj in range(len(pred)):
                    if gt_start <= pred[jj] and pred[jj] <= gt_end:
                        include_list.append(pred[jj])
                        is_include += 1
                if is_include == 0:
                    deletion += 1
                elif is_include > 1:
                    insertion = is_include - 1
                elif is_include == 1:
                    correct += 1
    if pred[0] == None:
        substitution = 0
    else:
        substitution = len(pred) - len(list(set(include_list)))
    return substitution, deletion, insertion, correct

def evaluation(gt_list, answer_list, print_result = True):
    set_num = len(gt_list)
    total_s, total_d, total_i, total_n, total_correct = 0, 0, 0, 0, 0
    for i in range(set_num):
        sw_s, sw_d, sw_i, sw_n, sw_correct = 0, 0, 0, 0, 0
        for j in range(3): # for drones 1 ~ 3
            dw_s, dw_d, dw_i, dw_n, dw_correct = 0, 0, 0, 0, 0
            for k in range(3): # for class man, woman, child
                cw_s, cw_d, cw_i, cw_correct = cw_metrics_cal(gt_list[i][j][k], answer_list[i][j][k])
                dw_s += cw_s
                dw_d += cw_d
                dw_i += cw_i
                dw_n += len(gt_list[i][j][k])
                dw_correct += cw_correct
            dw_er = (dw_s + dw_d + dw_i) / dw_n
            if print_result:
                print('Set', str(i), 'Drone', str(j), 's, d, i, er, correct:', dw_s, dw_d, dw_i, np.round(dw_er, 2), dw_correct)
                    
            sw_s += dw_s
            sw_d += dw_d
            sw_i += dw_i
            sw_n += dw_n
            sw_er = (sw_s + sw_d + sw_i) / sw_n
            sw_correct += dw_correct
        total_s += sw_s
        total_d += sw_d
        total_i += sw_i
        total_n += sw_n
        total_er = (total_s + total_d + total_i) / total_n
        total_correct += sw_correct
        if print_result:
            print('Subtotal Set', str(i), 's, d, i, er, correct:', sw_s, sw_d, sw_i, np.round(sw_er, 2), sw_correct)
    if print_result:
        print('Total', 's, d, i, er, correct:', total_s, total_d, total_i, np.round(total_er, 2), total_correct)
    return total_s, total_d, total_i, total_er, total_correct

def postprocessing(m, f, b, threshold=0.8, smooth=5, min_frame=3, merge_frame=10, visualize=False, save_path=None, gt=None):
    # smoothing
    m_smooth = moving_average(np.pad(m, (smooth//2,smooth//2), mode='edge'), n=smooth)
    f_smooth = moving_average(np.pad(f, (smooth//2,smooth//2), mode='edge'), n=smooth)
    b_smooth = moving_average(np.pad(b, (smooth//2,smooth//2), mode='edge'), n=smooth)
    # threshold
    m_threshold = np.asarray(m_smooth>threshold, dtype='int')
    f_threshold = np.asarray(f_smooth>threshold, dtype='int')
    b_threshold = np.asarray(b_smooth>threshold, dtype='int')
    # index 
    m_index = np.where(m_threshold==1)[0]
    f_index = np.where(f_threshold==1)[0]
    b_index = np.where(b_threshold==1)[0]
    # consecutive segment
    m_consecutive = consecutive(m_index)
    f_consecutive = consecutive(f_index)
    b_consecutive = consecutive(b_index)
    # merge if two segment close
    m_segment = consecutive_merge(m_consecutive, threshold=merge_frame)
    f_segment = consecutive_merge(f_consecutive, threshold=merge_frame)
    b_segment = consecutive_merge(b_consecutive, threshold=merge_frame)
    # middle point
    m_answer, m_frame = seg_to_answer(m_segment, m, min_frame)
    f_answer, f_frame = seg_to_answer(f_segment, f, min_frame)
    b_answer, b_frame = seg_to_answer(b_segment, b, min_frame)
    
    m_answer = np.asarray(m_answer)
    f_answer = np.asarray(f_answer)
    b_answer = np.asarray(b_answer)
    if visualize:
        plt.figure(figsize=(16,9))
        plt.figure(figsize=(16,9))
        plt.subplot(3,2,1)
        # librosa.display.specshow(torch.log(mixture_spec.clamp(min=1e-5)).detach().squeeze().cpu().numpy())
        # plt.title('log spec')
        plt.subplot(3,2,3)
        librosa.display.specshow(np.asarray([m, f, b]))
        plt.title('est prob')
        plt.subplot(3,2,5)
        librosa.display.specshow(np.asarray([m_smooth, f_smooth, b_smooth]))
        plt.title('est prob smooth')
        plt.subplot(3,2,2)
        librosa.display.specshow(np.asarray([m_threshold, f_threshold, b_threshold]))
        plt.title('est prob threshold={}'.format(threshold))
        plt.subplot(3,2,4)
        librosa.display.specshow(np.asarray([m_frame, f_frame, b_frame]))
        plt.title('est segment and mid point')
        if gt is not None:
            plt.subplot(3,2,6)
            gt_frame = np.zeros((3, len(m_frame)))
            for index, segments in enumerate(gt):
                for segment in segments:
                    start, end = segment
                    gt_frame[index, (int)(start*(44100/4096)):(int)(end*(44100/4096))] = 1
            librosa.display.specshow(gt_frame)
            plt.title('gt frame')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return m_answer, f_answer, b_answer

def process_answer(male_prob, female_prob, baby_prob, window=400, hop=200, threshold=0.5, smooth=5, min_frame=3, merge_frame=10, gt=None, save_path=None, visualize=True):
    male_smooth = moving_average(np.pad(male_prob, (smooth//2,smooth//2), mode='edge'), n=smooth)
    female_smooth = moving_average(np.pad(female_prob, (smooth//2,smooth//2), mode='edge'), n=smooth)
    baby_smooth = moving_average(np.pad(baby_prob, (smooth//2,smooth//2), mode='edge'), n=smooth)
    # threshold
    male_threshold = np.asarray(male_smooth>threshold, dtype='int')
    female_threshold = np.asarray(female_smooth>threshold, dtype='int')
    baby_threshold = np.asarray(baby_smooth>threshold, dtype='int')
    # index 
    male_index = np.where(male_threshold==1)[0]
    female_index = np.where(female_threshold==1)[0]
    baby_index = np.where(baby_threshold==1)[0]
    # consecutive segment
    male_consecutive = consecutive(male_index)
    female_consecutive = consecutive(female_index)
    baby_consecutive = consecutive(baby_index)
    # merge if two segment close
    male_segment = consecutive_merge(male_consecutive, threshold=merge_frame)
    female_segment = consecutive_merge(female_consecutive, threshold=merge_frame)
    baby_segment = consecutive_merge(baby_consecutive, threshold=merge_frame)
    # middle point
    male_answer, male_frame = seg_to_answer(male_segment, male_prob, min_frame)
    female_answer, female_frame = seg_to_answer(female_segment, female_prob, min_frame)
    baby_answer, baby_frame = seg_to_answer(baby_segment, baby_prob, min_frame)
    # plot
    male_answer = np.asarray(male_answer)
    female_answer = np.asarray(female_answer)
    baby_answer = np.asarray(baby_answer)
    if visualize:
        plt.figure(figsize=(16,9))
        plt.subplot(3,2,1)
        #librosa.display.specshow(torch.log(mixture_spec.clamp(min=1e-5)).detach().squeeze().cpu().numpy())
        plt.title('log spec')
        plt.subplot(3,2,3)
        logits_seq = np.stack([male_prob, female_prob, baby_prob], 0)
        librosa.display.specshow(logits_seq)
        plt.title('est prob')
        plt.subplot(3,2,5)
        librosa.display.specshow(np.asarray([male_smooth, female_smooth, baby_smooth]))
        plt.title('est prob smooth')
        plt.subplot(3,2,2)
        librosa.display.specshow(np.asarray([male_threshold, female_threshold, baby_threshold]))
        plt.title('est prob threshold={}'.format(threshold))
        plt.subplot(3,2,4)
        librosa.display.specshow(np.asarray([male_frame, female_frame, baby_frame]))
        plt.title('est segment and mid point')
        if gt is not None:
            plt.subplot(3,2,6)
            gt_frame = np.zeros((3, len(male_frame)))
            for index, segments in enumerate(gt):
                for segment in segments:
                    start, end = segment
                    gt_frame[index, (int)(start*(16000/1600)):(int)(end*(16000/1600))] = 1
            librosa.display.specshow(gt_frame)
            plt.title('gt frame')
        plt.tight_layout()
        os.makedirs(os.path.join(*save_path.split('/')[:-1]), exist_ok = True)
        plt.savefig(save_path)
        plt.close()
    return male_answer, female_answer, baby_answer

def validation(model, nowstr, ep):
    gt_list_for_new_samples = [[
        [[[187, 191], [194, 197]], [[199, 203]], [[51, 56], [202, 208]]],
        [[[147, 151], [162, 165]], [[154, 158]], [[143, 148]]],
        [[[197, 200], [204, 208]], [[198, 202]], [[210, 215]]],
    ]]
        
    set1_audio_list = glob.glob('../gc_2021_jh/validation_data_final/set01_drone01_ch*.wav')
    _, sr = librosa.load(set1_audio_list[0], sr=None, mono=True)
    audio1 = [librosa.load(path, sr=None, mono=True)[0] for path in set1_audio_list]
    audio1 = torch.from_numpy(np.asarray(audio1))
    audio1 = julius.resample_frac(audio1, sr, 16000)
    audio1 = torch.mean(audio1, axis=0).numpy()

    set2_audio_list = glob.glob('../gc_2021_jh/validation_data_final/set01_drone02_ch*.wav')
    _, sr = librosa.load(set1_audio_list[0], sr=None, mono=True)
    audio2 = [librosa.load(path, sr=None, mono=True)[0] for path in set2_audio_list]
    audio2 = torch.from_numpy(np.asarray(audio2))
    audio2 = julius.resample_frac(audio2, sr, 16000)
    audio2 = torch.mean(audio2, axis=0).numpy()

    set3_audio_list = glob.glob('../gc_2021_jh/validation_data_final/set01_drone03_ch*.wav')
    _, sr = librosa.load(set3_audio_list[0], sr=None, mono=True)
    audio3 = [librosa.load(path, sr=None, mono=True)[0] for path in set3_audio_list]
    audio3 = torch.from_numpy(np.asarray(audio3))
    audio3 = julius.resample_frac(audio3, sr, 16000)
    audio3 = torch.mean(audio3, axis=0).numpy()

    with torch.no_grad():
        model.reset_state()
        audio1 = torch.tensor(audio1).unsqueeze(0).unsqueeze(0).float().cuda()
        z1 = torch.sigmoid(model(audio1)).cpu().numpy().squeeze()
        m1, f1, b1 = z1[..., 0], z1[..., 1], z1[..., 2]
        model.reset_state()

        audio2 = torch.tensor(audio2).unsqueeze(0).unsqueeze(0).float().cuda()
        z2 = torch.sigmoid(model(audio2)).cpu().numpy().squeeze()
        m2, f2, b2 = z2[..., 0], z2[..., 1], z2[..., 2]
        model.reset_state()

        audio3 = torch.tensor(audio3).unsqueeze(0).unsqueeze(0).float().cuda()
        z3 = torch.sigmoid(model(audio3)).cpu().numpy().squeeze()
        m3, f3, b3 = z3[..., 0], z3[..., 1], z3[..., 2]

    result_dict = {}
    for threshold in np.linspace(0.1, 0.9, 30):
        for smooth in [3, 5, 7, 9, 11, 13, 15, 17]:
            for min_frame in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
                for merge_frame in [4, 8, 12, 16, 20]:
                    m1f, f1f, b1f = process_answer(m1, f1, b1, threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame, gt=gt_list_for_new_samples[0][0], save_path='./results/{}/valid_audio1_{}.png'.format(nowstr, ep), visualize=False)
                    m2f, f2f, b2f = process_answer(m2, f2, b2, threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame, gt=gt_list_for_new_samples[0][1], save_path='./results/{}/valid_audio2_{}.png'.format(nowstr, ep), visualize=False)
                    m3f, f3f, b3f = process_answer(m3, f3, b3, threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame, gt=gt_list_for_new_samples[0][2], save_path='./results/{}/valid_audio3_{}.png'.format(nowstr, ep), visualize=False)
                    answer_list = [[
                        [list(m1f), list(f1f), list(b1f)],
                        [list(m2f), list(f2f), list(b2f)],
                        [list(m3f), list(f3f), list(b3f)],
                    ]]
                    gt_list = [[
                        [[[187, 191], [194, 197]], [[199, 203]], [[51, 56], [202, 208]]],
                        [[[147, 151], [162, 165]], [[154, 158]], [[143, 148]]],
                        [[[197, 200], [204, 208]], [[198, 202]], [[210, 215]]],
                    ]]
                    s, d, i, er, correct = evaluation(gt_list, answer_list, False)
                    result_dict['{}_{}_{}_{}'.format(threshold, smooth, min_frame, merge_frame)] = er

    result_sorted = sorted(result_dict.items(), key=lambda x: x[1])
    print(list(filter(lambda x: x[1] == result_sorted[0][1], result_sorted)))
    print(result_sorted[0], result_sorted[-1])
    threshold, smooth, min_frame, merge_frame = np.asarray(result_sorted[0][0].split('_'), dtype='float')
    smooth = (int)(smooth)
    min_frame = (int)(min_frame)
    merge_frame = (int)(merge_frame)
    m1f, f1f, b1f = process_answer(m1, f1, b1, threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame, save_path = './results/{}/valid_audio1_{}.png'.format(nowstr, ep), gt=gt_list_for_new_samples[0][0])
    m2f, f2f, b2f = process_answer(m2, f2, b2, threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame, save_path = './results/{}/valid_audio2_{}.png'.format(nowstr, ep), gt=gt_list_for_new_samples[0][1])
    m3f, f3f, b3f = process_answer(m3, f3, b3, threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame, save_path = './results/{}/valid_audio3_{}.png'.format(nowstr, ep), gt=gt_list_for_new_samples[0][2])

    answer_list_for_sample = [
            [
                    [list(m1f), list(f1f), list(b1f)],
                    [list(m2f), list(f2f), list(b2f)],
                    [list(m3f), list(f3f), list(b3f)],
            ],
    ]

    total_s, total_d, total_i, total_er, total_correct = evaluation(gt_list_for_new_samples, answer_list_for_sample, True)
    print("best: threshold {} smooth {} min_frame {} merge_frame {}".format(threshold, smooth, min_frame, merge_frame))
    return dict(total_s=total_s, total_d=total_d, total_i=total_i, total_er=total_er, total_corret=total_correct, 
            threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame)

def test(model, threshold, smooth, min_frame, merge_frame):
    gt_list_for_new_samples = [[
        [[[187, 191], [194, 197]], [[199, 203]], [[51, 56], [202, 208]]],
    ]]
        
    test_audio_dir_2021 = 'test.wav'
    audio1, sr = sf.read(test_audio_dir_2021)
    audio1 = librosa.resample(audio1[:, 0], orig_sr=sr, target_sr=16000)

    with torch.no_grad():
        model.reset_state()
        audio1 = torch.tensor(audio1).view(1, 1, -1).float().cuda()
        z1 = torch.sigmoid(model(audio1)).cpu().numpy().squeeze()
        m1, f1, b1 = z1[..., 0], z1[..., 1], z1[..., 2]
        model.reset_state()

    m1f, f1f, b1f = process_answer(m1, f1, b1, threshold=threshold, smooth=smooth, min_frame=min_frame, merge_frame=merge_frame, gt=gt_list_for_new_samples[0][0], visualize=False)

    answer_list_for_sample = [
            [
                    [list(m1f), list(f1f), list(b1f)],
            ],
    ]
    print(answer_list_for_sample)

