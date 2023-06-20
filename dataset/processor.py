from .path_utils import get_paths_from_dirs, get
from pathlib import Path
import os
from dataset import PathLike
from abc import abstractmethod, ABC, ABCMeta
from pycocotools.coco import COCO
import pandas as pd
import json
import numpy as np
from shapely.geometry import Polygon
from .dataset import get_paths
import shutil


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
    def __init__(self, file, save_dir, tresh, is_more):
        super().__init__(file)
        self._save_dir = Path(save_dir)
        self._tresh = tresh
        self._is_more = is_more

    def process(self):
        labels = read_segmentation_labels(self.file)
        with open(self._save_dir / str(self.file.name), 'w') as f:
            for label in labels:
                mask = label[1:]
                p = Polygon([(x,y) for x,y in zip(mask[0::2], mask[1::2])] )

                if self._is_more:
                    if p.area > self._tresh:
                        f.write(self.__to_string(label))
                else:
                    if p.area < self._tresh:
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

def _todf(fjson):
    return pd.DataFrame(fjson)

def train_test_splite(path_2_image, path_2_anno):
    coco = COCO(path_2_anno)
    images = _todf(coco.imgs).T

class MergeAnnotation:
    """
        Merge annotation files from dirs to one json file using coco
    """
    def __init__(self, root_direcroty) -> None:

        self.path_to_annotations = list(filter( lambda x: True if x.split('.')[-1] == 'json' else False, get_paths(root_direcroty)))
        self.merge_annotations  = {}
        self.merge_images       = {}
        self.image_indexs       = {}

    def process(self):
        image_id = 1
        segment_id = 1
        for path in self.path_to_annotations:
            coco = COCO(path)
            for k, image in coco.imgs.items():  
                new_image = image.copy()
                old_name = image['file_name']
                new_name = self.__create_name(image['file_name'])
                new_image['file_name'] = new_name
                new_image['id'] = image_id
                self.merge_images[image_id] = new_image
                self.image_indexs[old_name] = {'id':image['id'],'new_id': image_id, 'new_name': new_image['file_name']}
                image_id+=1
                
            for k, ann in coco.anns.items():
                old_image_name = coco.imgs[ann['image_id']]['file_name']
                new_image_id = self.image_indexs[old_image_name]['new_id']
                new_annotation = ann.copy()
                new_annotation['id'] = segment_id
                new_annotation['image_id'] = new_image_id
                self.merge_annotations[segment_id] = new_annotation
                segment_id+=1
        pass
    
    def __create_name(self, name):
        assert isinstance(name, str), "name"
        parts = name.split('/')
        if len(parts) > 1:
            new_name = parts[-2] + "_" + parts[-1]
        else:
            new_name = name
        return new_name

    def save(self, out_dir, file_name):
        """
            Save merge files to json format
        """
        p = Path(out_dir)
        p_image = p / 'images'
        if p.is_dir():
            shutil.rmtree(out_dir)
            p.mkdir(parents=True, exist_ok = True)
            p_image.mkdir()
        else:
            p_image.mkdir() 

        coco = COCO(self.path_to_annotations[-1])
        categories = []
        for path in self.path_to_annotations:
            coco = COCO(path);
            for cat in coco.dataset['categories']:
                if not cat['name'] in [cat['name'] for cat in categories]:
                    categories.append(cat)
        res = {
                "info":        coco.dataset['info'],
                "licenses":    coco.dataset['licenses'],
                "categories":  categories,
                "images":      list(self.merge_images.values()),
                "annotations":list(self.merge_annotations.values())
              }
        with open(p / file_name, 'w') as f:
            json.dump(res, f)

        for path in self.path_to_annotations:
            coco = COCO(path)
            for k, image in coco.imgs.items():
                file_name = image['file_name']
                path_2_img = Path(path).parents[1] / 'images' / file_name
                new_file_name = self.image_indexs[file_name]['new_name']
                shutil.copy(path_2_img, p_image / new_file_name)


