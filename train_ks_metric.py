import sys
import os
sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from dataset.path_utils import get_files_from_dirs, get_paths_from_dirs
import pandas as pd
from yolov5.utils.general import xywhn2xyxy, xywh2xyxy, xyxy2xywh
import scipy.stats as stats
import subprocess

def get_diag(coords):
    if isinstance(coords, np.ndarray) and len(coords.shape) == 2:
        x1 = coords[0,0]
        y1 = coords[0,1]
        x2 = coords[0,2]
        y2 = coords[0,3]
    else:
        x1,y1,x2,y2 = coords
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

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



p = subprocess.Popen(['python', './yolov5/train.py', '--cfg', 'yolov5m.yaml','--data','../dataset/detection_set2/data_simple.yaml', 
                      '--imgsz', '512', '--name','ks', '--epoch', '5', '--exist-ok'])

p.wait()

log_train = {}
step = 5

for epoch in range(2,20):

    p = subprocess.Popen(['python', './yolov5/train.py', '--weights', '../yolov5/runs/train/ks/weights/last.pt',
                            '--data','../dataset/detection_set2/data_simple.yaml', 
                            '--imgsz', '512', '--name','ks', '--epoch',str(step), '--exist-ok'])
    p.wait()

    # !python ../../yolov5/train.py --weights ../../yolov5/runs/train/ks/weights/last.pt --data \
    # ../../../dataset/detection_set2/data_simple.yaml --imgs 512 --epoch {step} --name ks --batch-size 4 --exist-ok \
    # --project ../../yolov5/runs/train/

    p = subprocess.Popen(['python', './yolov5/detect.py', '--weights', '../yolov5/runs/train/ks/weights/last.pt',
                            '--source','../dataset/detection_set2/validation/', 
                            '--imgsz', '512', '--name','ks','--save-txt', '--exist-ok'])

    p.wait()

    predict_labels = get_bbox_size_arr('./yolov5/runs/detect/ks/labels/')
    train_labels   = get_bbox_size_arr('../dataset/detection_set2/validation/')
    r = ks_metric(predict_labels, train_labels)
    log_train[epoch*step] = r
    
frame_log = pd.DataFrame(log_train).T
frame_log.to_csv('ks_log.csv')

