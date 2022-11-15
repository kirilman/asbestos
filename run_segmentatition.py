import numpy as np
import os 
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from dataset import get_paths, PathLike, is_image
import pandas as pd
import shutil

import subprocess


path_2_test = Path('../dataset/segmentation/test/')

dataset_root = "/home/kirilman/Project/dataset/segmentation/test.yaml"
path_2_nets  = {'small': Path('./yolov5/runs/train-seg/small_l_b4_512/weights/best.pt'),
                'large': Path('./yolov5/runs/train-seg/large_l_b4_512/weights/best.pt'),
                'gener': Path('./yolov5/runs/train-seg/m_batch4_imgs512/weights/best.pt'),
                'generl':Path('./yolov5/runs/train-seg/l_batch4_imgs512/weights/best.pt') }

path_2_dataset = {'small': Path('../dataset/segmentation/labels/small/test/'),
                  'large': Path('../dataset/segmentation/labels/large_segment/test/'),
                  'gener': Path('../dataset/segmentation/labels/original/test/'),}

def copy(_from, to):
    files = os.listdir(_from)
    files = list(filter( lambda x:False if is_image(x) else True, files))
    for file in files:
        shutil.copy(_from / file, to / file)

with open('result.txt','w') as f:              
    for n, path_2_data in path_2_dataset.items():
        copy(path_2_data, path_2_test)
        for name, path_2_net in path_2_nets.items():
            print('dataset:*',n,'*', path_2_net,'*')
            # python ../../yolov5/segment/val.py --data {dataset_root} --weight {path_2_net} --project ./result/ --imgsz 512 --batch-size 1 --name {name} > r.txt

            p = subprocess.Popen(['python', './yolov5/segment/val.py', '--data',dataset_root, '--weight', path_2_net, '--imgsz', '512', '--batch-size', '1',
                                  '--iou-thres','0.5'], stdout=f)
            p.wait()

            # f.write(p.stdout)
            print('-------------------------------------------')
            
