__all__ = (
    "is_image",
    "load_img",
    "ImageDirDataset",
    "get_paths",
    "load_txt"
)

from abc import ABC
import sys, os
from functools import lru_cache

# sys.path.append(os.getcwd())

import numpy as np
from PIL import Image
import os 
from pathlib import Path
from typing import List, Dict, Union
import glob
from abc import ABC, abstractmethod
import cv2
from .flogging import debug_load_img
PathLike = Union[Path, str]

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp' 

def is_image(path:PathLike):
    path = str(path)
    return path.split('.')[-1].lower() in IMG_FORMATS

class Bbox():
    def __init__(self,x1,y1,x2,y2) -> None:
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        
    @property
    def diagonal(self):
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)



#Дебаг считавания (декоратор)
@debug_load_img
def load_img(filepath, dtype = float, convert_type = None)-> np.array:
    # if convert_type is not None:
    #     img = Image.open(filepath).convert(convert_type)
    # else:
    #     img = Image.open(filepath)
    img = cv2.imread(filepath)
    return np.array(img, dtype = dtype)
            
def load_txt(path:PathLike)->str:
    with open(path, 'r') as f:
        data = f.read()
    return data

def get_paths(path: PathLike)->List[PathLike]:
    if isinstance(path, str): 
        path = Path(path)
    files = []
    if path.is_dir():
        files += glob.glob(str(path / '**' / '*.*'), recursive=True)
    elif path.is_file():  # file
        with open(path) as t:
            t = t.read().strip().splitlines()
            parent = str(path.parent) + os.sep
            files += [x.replace('./', parent) if x.startswith('./') else x for x in t] 
    else:
        raise FileNotFoundError(f'{path} does not exist')         
    return files   

def get_subdirs(path: PathLike)->Dict[PathLike,int]:
    
    files = get_paths(path)
    s_pos = len(str(path))
    sub_dirs = set()
    count_images_by_subdir = {}
    for p in files:
        e_pos = str(p).rfind('/')
        sub_dirs.add(p[s_pos+1:e_pos])
    return sub_dirs

class AsbestosDataSet:
    def __init__(self, image_dir, mask_dir, transform = None, preload_data = False) -> None:
        self.image_names = os.listdir(image_dir) 
        self.mask_names = os.listdir(mask_dir)
        self.transform = transform
        self.preload_data = preload_data
        if len(self.image_names) != len(self.mask_names):
            print("Inside {} number of images and masks are different".format(image_dir))
            f_split = lambda x: x.split('.')[0]
            image_suffix = Path(self.image_names[0]).suffix
            mask_suffix  = Path(self.mask_names[0]).suffix
            intersection = list( set(map(f_split, self.image_names)).intersection(list(map(f_split,self.mask_names))))   
            self.image_names = [s + image_suffix for s in intersection]
            self.mask_names = [s + mask_suffix for s in intersection]
        self.number_images = len(self.image_names)
        self.image_names.sort(); self.mask_names.sort()
        self.image_paths = [Path(image_dir, name) for name in self.image_names]
        self.mask_paths = [Path(mask_dir, mask) for mask in self.mask_names]
        self.n = 0
        
        for image, mask in zip(self.image_paths, self.mask_paths):
            assert image.name.split('.')[0] == mask.name.split('.')[0], "Names do not match, {} and {}".format(image, mask)
            
        if self.preload_data:
            self.raw_images = [load_img(p, np.float32,"L")/255 for p in self.image_paths]
            self.raw_masks  = [load_img(p, np.float32, "L")/255 for p in self.mask_paths]

    def __len__(self):
        return self.number_images

    def __iter__(self):
        self.n = 0
        return self

    def __getitem__(self, index):
        try:
            if self.preload_data:
                img = self.raw_images[index]
                mask = self.raw_masks[index]
            else:    
                img  = load_img(self.image_paths[index], np.float32, "L" )/255
                mask = load_img(self.mask_paths[index], np.float32 ,"L")/255
            #FIX
#             mask = np.logical_not(mask).astype(np.long) 
            name = self.image_names[index]
            if self.transform:
                if not isinstance(self.transform, list):
                    operators = [self.transform]
                else:
                    operators = self.transform
                for operator in operators:
                    img = operator(image=img)['image']
                    mask = operator(image=mask)['image']
            return {'image': img, 'mask': mask, 'path': name}
        except:
            raise 
        
    def __next__(self):
        if self.n < self.number_images:
            res = self.__getitem__(self.n)
            self.n += 1
            return res
        else:
            raise StopIteration

class ImageDirDataset(dict):
    def __init__(self, path) -> None:
        self.path = path
        self.paths = get_paths(self.path)
        self.image_files = [p for p in self.paths if Path(p).is_file() and p.split('.')[-1] in IMG_FORMATS]
    
    def __getitem__(self, index):
        if isinstance(index,int):
            return {"name": self.image_files[index],"image":load_img(self.image_files[index])}
        elif isinstance(index, str):
            names = [Path(f).stem for f in self.image_files]
            possible_index = []
            for k, name in enumerate(names):
                if name == index:
                    possible_index.append(k)
            return [{"name": self.image_files[i], "image": load_img(self.image_files[i])} for i in possible_index]

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    ds = ImageDirDataset('/home/kirilman/Project/dataset/detection_set2/')
    print(ds.image_files)
    for f in ds.image_files:
        print(hash(f), f)
    print(ds[1])
   