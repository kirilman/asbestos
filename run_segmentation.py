import numpy as np
import os 
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from dataset import get_paths, PathLike, is_image
from dataset.path_utils import get_paths_from_dirs
import pandas as pd
import shutil
import subprocess
import argparse

def copy(_from, to):
    files = os.listdir(_from)
    files = list(filter( lambda x:False if is_image(x) else True, files))
    for file in files:
        shutil.copy(_from / file, to / file)

def run_segmentation():
    path_2_test = Path('../dataset/segmentation/test/')

    dataset_root = "/home/kirilman/Project/dataset/segmentation/test.yaml"
    path_2_nets  = {'small': Path('./yolov5/runs/train-seg/small_l_b4_512/weights/best.pt'),
                    'large': Path('./yolov5/runs/train-seg/large_l_b4_512/weights/best.pt'),
                    'gener': Path('./yolov5/runs/train-seg/m_batch4_imgs512/weights/best.pt'),
                    'generl':Path('./yolov5/runs/train-seg/l_batch4_imgs512/weights/best.pt') }

    path_2_dataset = {'small': Path('../dataset/segmentation/labels/small/test/'),
                    'large': Path('../dataset/segmentation/labels/large_segment/test/'),
                    'gener': Path('../dataset/segmentation/labels/original/test/'),}

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
                
def run_validation_for_model(path, path_test_yml):
    if isinstance(path,str):
        path = Path(path)
    models = get_paths_from_dirs([path],['pt'])
    models = list(filter(lambda x: x.name.split('/')[-1] == 'best.pt', models))
    print(len(models), models[-1])
    for path_2_model in models:
        name = str(path_2_model).split('/')[-3]
        if task == 'val_dir':
            p = subprocess.Popen(['python', './yolov5/segment/val.py', '--data', path_test_yml, '--weight', path_2_model, '--imgsz', '512','--batch-size', '1', '--name', name])
        else:
            p = subprocess.Popen(['python', './yolov5/segment/predict.py', '--source', path_test_yml, '--weight', path_2_model, '--imgsz', '512','--name', name])

        p.wait()
        print('-----\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--task', type = str, help="Task for run: val_dir")
    parser.add_argument('--path', type=str, help='Path to dir with models')
    parser.add_argument('--path_yaml', type=str, help='Path with yaml dataset')
    args = parser.parse_args()
    task = args.task
    path_2_model = args.path
    path_2_config = args.path_yaml
    if task == 'val_dir':
        run_validation_for_model(path_2_model, path_2_config)
    elif task =='pred':
        run_validation_for_model(path_2_model, path_2_config)
