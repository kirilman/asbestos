from json import load
from sklearn.datasets import load_sample_image
from dataset import load_img
from utils import get_paths_from_dirs
import cv2
from typing import List, Tuple
import os
from pathlib import Path
import numpy as np

def resize_images(inpt_dir: List, formats: List, out_dir: str, out_size: Tuple):
    paths = get_paths_from_dirs(inpt_dir, formats)
    assert len(paths) > 0, "Files haven't found"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,)
    out_dir = Path(out_dir)
    for p in paths:
        img = load_img(p)
        img = cv2.resize(img, out_size)
        cv2.imwrite(str(out_dir.joinpath(p.name)), img)

def transform_images(inpt_dir: List, formats: List, out_dir: str, transform_compose, kwargs):
    paths = get_paths_from_dirs(inpt_dir, formats)
    assert len(paths) > 0, "Files haven't found"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,)
    out_dir = Path(out_dir)

    for t in range(kwargs['times']):
        for p in paths:
            img = load_img(p)
            augmented = transform_compose(image=img)
            cv2.imwrite(str(out_dir.joinpath(p.name)) + "_{}".format(t), augmented['image'])

def transform(image_dir, mask_dir, save_dir, transform, **kwargs):
    image_parts = sorted(get_paths_from_dirs([image_dir],['*']))
    mask_parts  = sorted(get_paths_from_dirs([mask_dir], ['*']))
    assert len(image_parts) == len(mask_parts), "Different counts"
    times = kwargs["times"]

    print(times)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir+"/image")
        os.makedirs(save_dir+"/segmentation") 
    save_dir = Path(save_dir)

    for t in range(times):
        for im_p, msk_p in zip(image_parts, mask_parts):
            image = load_img(im_p, int)
            mask = load_img(msk_p, int)
            transformed = transform(image=image, mask=mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            cv2.imwrite(str(save_dir.joinpath('image').joinpath("{}_{}".format(t, im_p.name))),
                        transformed_image)
            cv2.imwrite(str(save_dir.joinpath("segmentation").joinpath("{}_{}".format(t, msk_p.name))),
                        transformed_mask)
