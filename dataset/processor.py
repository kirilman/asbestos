# from .path_utils import get_paths_from_dirs
from pathlib import Path
import os
from dataset import PathLike
from abc import abstractmethod, ABC
from pycocotools.coco import COCO
import pandas as pd
import json
import numpy as np


class Extractor:
    def __init__(path_for_processing: PathLike):
        paths = get_paths_from_dirs()


class FileProcessing(ABC):
    def __init__(self, file):
        self.file = file
        pass

    @abstractmethod
    def process(self):
        pass

def create_segmentation_frame(data):
    return pd.DataFrame({d['id']: d for d in data['annotations']}).T

def get_segmentation_from_frame(frame: pd.DataFrame, img_id: int, category_id:int, images):
    maska = (frame.image_id == img_id) & (frame.category_id == category_id)
    h_img = images[img_id]['height']
    w_img = images[img_id]['width']
    return [{category_id: normalize_segment(np.array(list(s[0])), h_img, w_img)} for s in frame[maska].segmentation]

def normalize_segment(segment, h_img, w_img):
    segment[::2] /=w_img
    segment[1::2]/=h_img
    return segment


class JsonSegmentProcessing(FileProcessing):
    def __init__(self, file, category_id, save_dir):
        super().__init__(file)
        self._category_id = category_id
        self._save_dir = Path(save_dir)

    def process(self):
        with open(self.file, 'r') as file:
            data = json.load(file)

        frame = create_segmentation_frame(data)
        images_names = {f['id']:f['file_name'] for f in data['images']}
        #cЧИТАТЬ ОБЪЕКТ, ПОЛОЖИТЬ ИМЕНА В dict
        for img_id in images_names.keys():
            segmentation = get_segmentation_from_frame(frame, img_id, self._category_id, data['images'])
            save_path = str(self._save_dir / images_names[img_id].split('/')[-1].split('.')[-2]) + '.txt'
            self._save_segmentation(segmentation, save_path)
            print(img_id, save_path)
        return segmentation

    def _save_segmentation(self, segmentation, save_path):
        with open(save_path, 'w') as file:
            for segment in segmentation:
                for cl_obj, coords in segment.items():
                    file.write('{} '.format(cl_obj))
                    for c in coords:
                        file.write('{0:.4f}'.format(c)+' ')            
                file.write('\n')

file_name = '/home/kirilman/Project/dataset/detection_set/coco_labels/pits_stones_detections_161120.json'
proccesor = JsonSegmentProcessing(file_name, 1, '/home/kirilman/Project/asbestos/notebooks/test/')
proccesor.process()
