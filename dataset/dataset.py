import numpy as np
from PIL import Image
import os 
from pathlib import Path
from typing import List, Dict


class AsbestosDataSet:
    def __init__(self, image_dir, mask_dir, transform = None) -> None:
        self.image_names = os.listdir(image_dir) 
        self.mask_names = os.listdir(mask_dir)
        self.transform = transform
        if len(self.image_names) != len(self.mask_names):
            print('Number of images and masks are different')
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
            assert image.name.split('.')[0] == mask.name.split('.')[0], 'Names do not match, {} and {}'.format(image, mask)

    def __len__(self):
        return self.number_images

    def __iter__(self):
        self.n = 0
        return self

    def __getitem__(self, index):
        try:
            img = np.array(Image.open(self.image_paths[index]).convert('L'), dtype = np.float32)/255
            mask = np.array(Image.open(self.mask_paths[index]).convert('L')) // 255
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
            
    def union(self, other: "AsbestosDataSet")->"AsbestosDataSet":
        new_dataset = AsbestosDataSet('.','.')
        new_dataset.image_paths = list(set(self.image_paths+other.image_paths))
        new_dataset.mask_paths = list(set(self.mask_paths+other.mask_paths))
        new_dataset.image_names = list(set(self.image_names+other.image_names))
        new_dataset.number_images = len(new_dataset.image_names)
        return new_dataset

