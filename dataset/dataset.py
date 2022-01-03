import numpy as np
from PIL import Image
import os 
from pathlib import Path


class AsbestosDataSet:
    def __init__(self, image_dir, mask_dir) -> None:
        self.image_names = os.listdir(image_dir) 
        self.mask_names = os.listdir(mask_dir)
        assert len(self.image_names) == len(self.mask_names), 'Number of images and masks are different'
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
            img = Image.open(self.image_paths[index])
            mask = Image.open(self.mask_paths[index]).convert('L')
            name = self.image_names[index]
            return {'image': img, 'mask': mask, 'path': name}
        except:
            raise 
        
    def __next__(self):
        if self.n < self.number_images:
            img = Image.open(self.image_paths[self.n])
            mask = Image.open(self.mask_paths[self.n]).convert('L')
            name = self.image_names[self.n]
            self.n += 1
            return {'image': img, 'mask': mask, 'path': name}
        else:
            raise StopIteration
