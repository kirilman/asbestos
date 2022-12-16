import sys
import os
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLOv5 root directory
print(list(FILE.parents))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
print(sys.path)
from dataset.path_utils import get_paths_from_dirs
import numpy as np
from yolov7.utils.general import xywh2xyxy
from scipy.stats import kstest, wasserstein_distance
import subprocess
import shutil

def bbox_diagonal(coords):
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
            bbox_sizes.append(bbox_diagonal(box))
    return np.array(bbox_sizes)

def ks_metric(a,b):
    r = kstest(a, b)
    return {'statistic': r.statistic, 'pvalue': r.pvalue}

def predict_val(save_dir, epoch, epochs, last_weight, val_path, imgsz, option):
    """
        Predict validation images by current model and calculate metric
        Returns:
            r: float metric value
    """
    save_dir_labels = save_dir / 'labels'
    predict_labels = get_bbox_size_arr(save_dir_labels)
    train_labels   = get_bbox_size_arr(str(val_path))
    print(len(predict_labels), len(train_labels))
    r = wasserstein_distance(predict_labels, train_labels)
    if epoch < epochs - 1:
        files_labels = os.listdir(save_dir_labels)
        for f in files_labels:
            os.remove(save_dir_labels / Path(f))
    return r