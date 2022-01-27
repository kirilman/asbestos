import numpy as np
from PIL import Image
import os 
from pathlib import Path
from typing import List, Dict

def load_img(filepath, dtype, convert_type = None)-> np.array:
    if convert_type is not None:
        img = Image.open(filepath).convert(convert_type)
    else:
        img = Image.open(filepath)
    return np.array(img, dtype = dtype)

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
            self.raw_images = [load_img(p, np.float32, 'L')/255 for p in self.image_paths]
            self.raw_masks  = [load_img(p, np.float32, 'L')//255 for p in self.mask_paths]

    def __len__(self):
        return self.number_images

    def __iter__(self):
        self.n = 0
        return self

    def __getitem__(self, index):
        try:
            if self.preload_data:
                img = self.raw_images[index]
                mask = self.raw_images[index]
            else:    
                img  = load_img(self.image_paths[index], np.float32, 'L')/255
                mask = load_img(self.image_paths[index], np.float32, 'L')//255
            mask = np.logical_not(mask).astype(np.long)
            name = self.image_names[index]
            if self.transform:
                transformed = self.transform(image=img, mask=mask)
                img = transformed['image']
                mask= transformed['mask']
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
            