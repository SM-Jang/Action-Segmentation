import pdb
import numpy as np
import argparse


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content

def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends

def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float64)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1, #deletion cost
                              D[i, j-1] + 1, # insertion cost
                              D[i-1, j-1] + 1) # substitution cost
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]

    return score

def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class) 
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)

def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
    
    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        # Element-wise minimum(or maximum) of array elements.
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        
        # Get the best scoring segment
        idx = np.array(IoU).argmax() 

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def seg_eval(dataset, split, op):

    result = dict()
    recog_path        = "./results/{}/split_{}/{}/".format(dataset, split, op)
    ground_truth_path = "../data/{}/groundTruth/".format(dataset)
    file_list         = "../data/{}/splits/test.split{}.bundle".format(dataset, split)
    

    list_of_videos = read_file(file_list).split('\n')[:-1] 
    

    overlap = [.1, .25, .5, .75]
    tp, fp, fn = np.zeros(4), np.zeros(4), np.zeros(4)

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid
        gt_content = read_file(gt_file).split('\n')[0:-1]
        recog_file = recog_path + vid
        recog_content = read_file(recog_file).split('\n')[1].split()
        

        for i in range(len(gt_content)):  
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1 # True positive
            fp[s] += fp1 # False positive
            fn[s] += fn1 # False negative
            
    print('Acc: {:.4f}'.format(100*float(correct)/total))
    print('Edit: {:.4f}'.format((1.0*edit)/len(list_of_videos)))
    result['Acc']  = '{:.4f}'.format(100*float(correct)/total)
    result['Edit'] =  '{:.4f}'.format((1.0*edit)/len(list_of_videos))
    for s in range(len(overlap)):
        # 10%, 25%, 50%, 75% segment
        precision = tp[s] / float(tp[s]+fp[s]) 
        recall = tp[s] / float(tp[s]+fn[s])
        f1 = 2.0 * (precision*recall) / (precision+recall)
        f1 = np.nan_to_num(f1)*100
        print('F1@{:.2f}: {:.4f}'.format(overlap[s], f1))
        result[f'F1@{overlap[s]}'] = '{:.4f}'.format(f1)
        
        
    return result