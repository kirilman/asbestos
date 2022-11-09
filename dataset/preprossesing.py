from json import load
from sklearn.datasets import load_sample_image
from dataset import load_img
from .path_utils import get_paths_from_dirs
import cv2
from typing import List, Tuple,Dict
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def resize_images(inpt_dir: List, formats: List, out_dir: str, out_imag_size: Tuple, kwargs: Dict):
    if "dtype" in kwargs: 
        dtype = kwargs["dtype"]
    else:
        dtype = float
    paths = get_paths_from_dirs(inpt_dir, formats)
    assert len(paths) > 0, "Files haven't found"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir,)
    out_dir = Path(out_dir)
    for p in paths:
        img = load_img(str(p), dtype)
        img = cv2.resize(img, out_imag_size)
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
            assert transformed_image.sum() > 0, "Bad tranform result"
            cv2.imwrite(str(save_dir.joinpath('image').joinpath("{}_{}".format(t, im_p.name.replace('png','jpg')))),
                        transformed_image)
            cv2.imwrite(str(save_dir.joinpath("segmentation").joinpath("{}_{}".format(t, msk_p.name))),
                        transformed_mask)

#Create asbest segmentation from annotation file
def create_segmentation_from_annotation(coco_dataset, image_root, out_segmentation_root, category_name):
    for idx, image_name in coco_dataset.imgs.items():
        name = image_name['file_name'].split('/')[-1].split('.')[0]
        try:
            mask = create_annotation_mask(coco_dataset, idx, image_root, category_name)
            plt.imsave(os.path.join(out_segmentation_root, name + '.png'), mask, cmap='Greys')
        except Exception as err:
            print(idx, err)       
        if idx%10 == 0:
            print(idx)

def create_annotation_mask(coco_set, img_indx, image_root, category_name = 'asbest'):
    cat_ids  = coco_set.getCatIds(catNms=[category_name])
    img_ids  = coco_set.getImgIds(catIds=cat_ids );
    image_dict = coco_set.loadImgs(img_indx)[0]
    ann_ids = coco_set.getAnnIds(imgIds=image_dict['id'], catIds=cat_ids, iscrowd=None)
    anns = coco_set.loadAnns(ann_ids)
    if len(anns) > 0:
        for i in range(len(anns)):
            if i == 0:
                mask = np.array(coco_set.annToMask(anns[0]), dtype = np.int64)
            else:
                mask += coco_set.annToMask(anns[i])

        mask[mask>1] = 1#for stones!
    else: 
        w = coco_set.loadImgs(1)[0]['width']
        h = coco_set.loadImgs(1)[0]['height']
        mask = np.zeros((h,w))
    return mask