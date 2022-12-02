import sys
import os
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataset.path_utils import get_files_from_dirs, get_paths_from_dirs
import pandas as pd
import scipy.stats as stats
import subprocess
import torch
import numpy

def get_diag(coords):
    if isinstance(coords, np.ndarray) and len(coords.shape) == 2:
        x1 = coords[0,0]
        y1 = coords[0,1]
        x2 = coords[0,2]
        y2 = coords[0,3]
    else:
        x1,y1,x2,y2 = coords
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def get_bbox_size_arr(path_2_label, image_names = None):
    labels_files = get_paths_from_dirs([path_2_label],['txt'])
    if image_names:
        labels_files = list(filter(lambda x: True if str(x.name).split('.')[0] in image_names else False, labels_files))
    bbox_sizes = []
    for p in labels_files:
        data = np.loadtxt(p)
        if len(data.shape) > 1:
            bboxs = data[:,1:5]
        else:
            bboxs = data[1:5].reshape(1,-1)
        for i,box in enumerate(xywh2xyxy(bboxs)):
            bbox_sizes.append(get_diag(box))
    return np.array(bbox_sizes)

def ks_metric(a,b):
    r = stats.kstest(a, b)
    return {'statistic': r.statistic, 'pvalue': r.pvalue}


p = subprocess.Popen(['python', './yolov5/train.py', '--weights', 'yolov5m.pt','--data','../dataset/detection_set2/data_simple.yaml', 
                      '--imgsz', '512', '--name','ks', '--epoch', '3', '--exist-ok', '--batch-size','4'])

p.wait()

log_train = {}

with open('log_train.txt', 'w') as f:
    for epoch in range(10):

        p = subprocess.Popen(['python', './yolov5/train.py', '--weights', './yolov5/runs/train/ks/weights/last.pt',
                                '--data','../dataset/detection_set2/data_simple.yaml', 
                                '--imgsz', '512', '--name','ks', '--epoch','3', '--exist-ok', '--batch-size','4'])
        p.wait()

        p = subprocess.Popen(['python', './yolov5/detect.py', '--weights', './yolov5/runs/train/ks/weights/last.pt',
                                '--source','../dataset/detection_set2/validation/', 
                                '--imgsz', '512', '--name','ks','--save-txt', '--exist-ok'])

        p.wait()

        predict_labels = get_bbox_size_arr('./yolov5/runs/detect/ks/labels/')
        train_labels   = get_bbox_size_arr('../dataset/detection_set2/validation/')
        
        r = ks_metric(predict_labels, train_labels)
        log_train[epoch] = r

        f.write("{}:{}:{}\n".format(epoch*3, r['statistic'], r['pvalue']))
        print(epoch)
frame_log = pd.DataFrame(log_train).T
frame_log.to_csv('ks_log.csv')

