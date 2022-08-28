import imp
from typing import Tuple
import numpy as np
import os
import glob
import cv2
import albumentations as A
import random
from dataset.dataset import load_img
from pathlib import Path
from tqdm import tqdm

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self):
        self.transform = None
        T = [
            A.MedianBlur(p=0.5),
            A.ToGray(p=0.5),
            A.CLAHE(p=0.5),
            A.RandomBrightnessContrast(contrast_limit = 0.05, p=0.5),
            A.RandomGamma(p=0.5),
            A.VerticalFlip(p = 0.5),
            A.Flip(p = 0.5)
            # A.ImageCompression(quality_lower=0.9, p=0.1)
            ]  # transforms
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __call__(self, im, labels, p=1.0):
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])])
        return im, labels

class yolo_image_generator():
    def __init__(self, 
                 path,
                 img_size: Tuple,
                 max_count: int,
                 ):
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.max_count = max_count
        files = sorted(glob.glob(os.path.join(path, '*.*')))
        self.im_files = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        self.labels_files = img2label_paths(self.im_files)
        self.albumentations = Albumentations()

    def __iter__(self):
        self.count = 0
        self.current_gen_image = 0
        return self

    def __len__(self):
        return len(self.im_files)

    def __next__(self):
        if self.current_gen_image > self.max_count:
            raise StopIteration
        if self.count >= self.__len__():
            self.count = 0
        path = self.im_files[self.count]
        img = cv2.imread(path)
        labels = np.loadtxt(self.labels_files[self.count], dtype=np.float64)
        if labels.any() <= 0:
            print("Negative yolo coordinate")
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_AREA)
        self.count+=1
        self.current_gen_image+=1
        try:
            img, labels = self.albumentations(img, self.__fix_bboxs(labels))
        except:
            print(path)
        return img, labels

    def __fix_bboxs(self, bboxs):
        "Reduce size bbox"
        if len(bboxs) == 1:
            bboxs = np.expand_dims(bboxs,0)   
        for i in range(len(bboxs)):
            bboxs[i][3] = np.abs(bboxs[i][3] - 0.5 / self.img_size[0])
            bboxs[i][4] = np.abs(bboxs[i][4] - 0.5 / self.img_size[1])
        return bboxs
            
def generate_dataset(path_2_data,
                     path_2_save,
                     count_images,
                     image_size):
    path_2_save = Path(path_2_save) if isinstance(path_2_save, str) else path_2_save
    image_gen = yolo_image_generator(path_2_data, image_size, count_images)
    for i, (image, labels) in tqdm(enumerate(iter(image_gen))):
        cv2.imwrite(path_2_save / "{}.jpg".format(i), image)
        np.savetxt( path_2_save / "{}.txt".format(i),labels)


