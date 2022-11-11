from .path_utils import get_paths_from_dirs
from pathlib import Path
import os
from dataset import PathLike
from abc import abstractmethod, ABC, ABCMeta
from pycocotools.coco import COCO
import pandas as pd
import json
import numpy as np
from shapely.geometry import Polygon

class Extractor:
    def __init__(path_for_processing: PathLike):
        paths = get_paths_from_dirs()


class FileProcessing(metaclass = ABCMeta):
    def __init__(self, file):
        self.file = file
        pass

    @abstractmethod
    def process(self):
        pass

def read_segmentation_labels(p: PathLike):
    with open(p, 'r') as f:
        lines = f.readlines()
        return [np.fromstring(line, sep=' ') for line in lines]

def create_segmentation_frame(data):
    return pd.DataFrame({d['id']: d for d in data['annotations']}).T

def get_segmentation_from_frame(frame: pd.DataFrame, img_id: int, category_id:int, image):
    maska = (frame.image_id == img_id) & (frame.category_id == category_id)
    h_img = image['height']
    w_img = image['width']
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
        self._debug = False

    def process(self):
        with open(self.file, 'r') as file:
            data = json.load(file)

        frame = create_segmentation_frame(data)
        images_names = {f['id']:{'file_name':f['file_name'],'width':f['width'], 'height':f['height']} for f in data['images']}
        for img_id, image in images_names.items():
            segmentation = get_segmentation_from_frame(frame, img_id, self._category_id, image)
            save_path = str(self._save_dir / images_names[img_id]['file_name'].split('/')[-1].split('.')[-2]) + '.txt'
            self._save_segmentation(segmentation, save_path)
            if self._debug:
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


class SegmentSquareFilter(FileProcessing):
    """
        tresh: float 0.005 
    """
    def __init__(self, file, save_dir, tresh):
        super().__init__(file)
        self._save_dir = Path(save_dir)
        self._tresh = tresh

    def process(self):
        labels = read_segmentation_labels(self.file)
        with open(self._save_dir / str(self.file.name), 'w') as f:
            for label in labels:
                mask = label[1:]
                p = Polygon([(x,y) for x,y in zip(mask[0::2], mask[1::2])] )
                if p.area > self._tresh:
                    f.write(self.__to_string(label))

    def __to_string(self, arr):
        s = ""
        for i,x in enumerate(arr):
            if i == 0:
                s+=str(int(x))+" "
            else:
                s+=str(np.round(x,4))+" "
        s = s[:-1] + "\n"
        return s
