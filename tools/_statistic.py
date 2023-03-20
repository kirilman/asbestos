from os.path import isdir
import numpy as np
import pandas as pd
import os, sys

sys.path.append(os.getcwd())
from ASBEST_VEINS_LABELING.labelutilits._path import list_ext
from ASBEST_VEINS_LABELING.labelutilits._coco_func import coco_annotations
from pycocotools.coco import COCO
from pathlib import Path
from typing import List, Union
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = ["collect_bbox_maxsize", "collect_segmentation_maxsize", "plot_hist", 
           "collect_maxsize_from_json", "collect_height_weight_mean_bbox"]

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y
    
def yolo2coco(xc, yc, w, h, image_width, image_height):
    xc, w = xc*image_width,  w*image_width
    yc, h = yc*image_height, h*image_height
    xmin = xc - w//2
    ymin = yc - h//2
    return xmin,ymin,w, h

def diagonal(coords):
    if isinstance(coords, np.ndarray) and len(coords.shape) == 2:
        x1 = coords[0,0]
        y1 = coords[0,1]
        x2 = coords[0,2]
        y2 = coords[0,3]
    else:
        x1,y1,x2,y2 = coords
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def distance(p1,p2):
    x0 = p1[0] - p2[0]
    y0 = p1[1] - p2[1]
    return np.sqrt(x0 ** 2 + y0 ** 2)

# def max_value(p1, p2):
#     x = abs(p1[0] - p2[0])
#     y = abs(p1[1] - p2[1])
#     return max(x, y)

def max_box_value(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return max(dx, dy)

def collect_bbox_maxsize(path_2_label, image_names = None):
    """
        Collect box max size from directroy with .txt coco files 
        Return:
            np.ndarray: bbox_sizes 
    """
    labels_files = list_ext(path_2_label,'txt')
    # labels_files = list(filter(lambda x: True if x.split('.')[0] in image_names else False, labels_files))    
    bbox_sizes = []
    for p in labels_files:
        data = np.loadtxt(Path(path_2_label) / p)
        if len(data.shape) > 1:
            bboxs = data[:,1:5]
        else:
            bboxs = data[1:5].reshape(1,-1)
        for i,box in enumerate(xywh2xyxy(bboxs)):
            bbox_sizes.append(max_box_value(box[0], box[1], box[2], box[3]))
    return np.array(bbox_sizes)

def collect_segmentation_maxsize(segment_file: str, image_names: List = None):
    """
        segment_file: str Path to dataset json format
        image_names: List contains names used files
    """
    coco = COCO(segment_file)
    frame = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    print(df_image.shape)
    if image_names:
        df_image = df_image[df_image.file_name.apply(lambda x: Path(x).stem in list(image_names))]

    image_names = [p.split('/')[-1].split('.')[0] for p in list(df_image.file_name)]
    image_dict = df_image.T.to_dict()
    ids = [img['id'] for img in image_dict.values()]
    frame = frame[frame.image_id.isin(ids)]
    arr_max_size = []
    for k, row in enumerate(frame.iterrows()):
        segment = np.array(row[1].segmentation)
        image_id = row[1].image_id
        if not image_id in image_dict:
            print('Is not predict labels for image {}'.format(image_id))
            continue    
        x_coords = segment[0][::2]
        y_coords = segment[0][1::2]
        IMAGE_W = image_dict[image_id]['width']
        IMAGE_H = image_dict[image_id]['height']
        x_coords = x_coords/IMAGE_W
        y_coords = y_coords/IMAGE_H
        Points = [(x,y) for x,y in zip(x_coords, y_coords)]
        max_distance = 0
        for p1 in Points:
            for p2 in Points:
                max_distance = max(max_distance, distance(p1,p2))
        if max_distance > 1:    
            max_distance = 1
        arr_max_size.append(max_distance)
    return np.array(arr_max_size)

def collect_height_weight_mean_bbox(segment_file: str, normalize: bool = True)-> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
        Return h,w, mean
    """
    coco = COCO(segment_file)
    df = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    image_dict = df_image.T.to_dict()

    arr_height = []
    arr_weight = []
    arr_mean_hw = []
    for k, row in enumerate(df.iterrows()):
        bbox = np.array(row[1].bbox)
        img_id = row[1].image_id  
        IMAGE_W = image_dict[img_id]['width']
        IMAGE_H = image_dict[img_id]['height']
        xl, yl, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        if normalize:
            w /=IMAGE_W
            h /=IMAGE_H
        arr_height.append(h)
        arr_weight.append(w)
        arr_mean_hw.append((h+w)/2)
    return np.array(arr_height), np.array(arr_weight), np.array(arr_mean_hw)

 # max_distance = max(max_distance, max_box_value(p1[0], p1[1], p2[0], p2[1]))

def collect_maxsize_from_json(segment_file, image_names = None):
    coco = COCO(segment_file)
    frame = pd.DataFrame(coco.anns).T
    df_image = pd.DataFrame(coco.imgs).T
    if image_names is None:
        image_names = [p.split('/')[-1].split('.')[0] for p in list(df_image.file_name)]
    df_image = df_image[df_image.file_name.apply(lambda x: Path(x).stem in list(image_names))]
    arr_res = []
    image_dict = df_image.T.to_dict()
    for k, row in enumerate(frame.iterrows()):
        image_id = row[1].image_id      
        bbox = np.array(row[1].bbox)
        if not image_id in image_dict:
            continue
        IMAGE_W = image_dict[image_id]['width']
        IMAGE_H = image_dict[image_id]['height']
        xl, yl, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        x1 = xl/IMAGE_W
        y1 = yl/IMAGE_H
        x2 = (xl + w)/IMAGE_W
        y2 = (yl + h)/IMAGE_H
        arr_res.append(max_box_value(x1, y1, x2, y2))
    return np.array(arr_res)

class Statistic:
    def __init__(self, path2anno, path2pred) -> None:
        self.path2anno = path2anno
        self.path2pred = path2pred
        if Path(self.path2pred).is_dir():
            self.predict_names = [Path(x).stem for x in list_ext(self.path2pred)]
        self.arr = None
        self.arr_other = None

    def test(self, image_names = None):
        diagonal_arr = collect_bbox_maxsize(self.path2pred)
        # max_size_arr = collect_segmentation_maxsize(self.path2anno, image_names)
        max_size_arr = collect_maxsize_from_json(self.path2anno, image_names)
        result = {}
        result['kstest'] = stats.kstest(diagonal_arr, max_size_arr)
        result['mannwhitneyu'] = stats.mannwhitneyu(diagonal_arr, max_size_arr)
        self.arr = diagonal_arr
        self.arr_other = max_size_arr
        return result

    def _hist(self):
        fig = plt.figure(dpi = 150, figsize=(11,5))
        ax = fig.gca()
        n = int(0.1 * len(self.arr))
        sns.histplot(self.arr,  kde = True,color='red' ,  alpha = 1,  label = 'Box', lw=3, ax= ax)
        sns.histplot(self.arr_other,  kde = True,color='black', alpha = 0.6,  label = 'Segmentation size', lw=3, ax= ax)
        ax.set_xlim(0,1)
        plt.grid()
        plt.show()
        plt.legend()
        return fig

    def _pdf(self, save_name = "01"):
        fig = plt.figure(dpi = 120)
        ax = sns.ecdfplot(self.arr,       color= 'grey',  label = 'Максимальный размер рамки',)
        ax = sns.ecdfplot(self.arr_other, color = 'black',label = 'Максимальный размер экземплярной разметки', )
        ax.legend(loc = 1)
        plt.xlabel("Размер")
        plt.ylabel("Количество")
        plt.title("")
        plt.savefig("{}".format(save_name))

    def __str__(self) -> str:
        return f"{len(self.arr)}, {len(self.arr_other)}"

def plot_hist(box_sizes, segmt_sizes, save_name):

    fig,ax = plt.subplots(1,2, dpi = 150, figsize = (15,5))
    sns.ecdfplot(box_sizes,       color= 'red',  label = 'Макс. размер рамки', ax = ax[0])
    ax[0] = sns.ecdfplot(segmt_sizes, color = 'green',label = 'Макс. экземплярной\n разметки', ax = ax[0])
    ax[0].legend(loc = 1)

    n = int(0.1 * len(box_sizes))
    sns.histplot(box_sizes, kde = True,color='red' ,  alpha = 1,  label = 'Макс. размер рамки', lw=3, ax= ax[1])
    sns.histplot(segmt_sizes,  kde = True,color='green', alpha = 0.6,  label = 'Макс. экземплярной\n разметки', lw=3, ax= ax[1])
    ax[1].legend(loc = 1)
    ax[1].set_xlim(0,1)
    plt.xlabel("Размер")
    plt.ylabel("Количество")
    plt.title("")
    plt.savefig(save_name)
    return fig

if __name__ == "__main__":
    ans = {}
    ans_ks = {}
    for p_fold in [p_fold for p_fold in Path("/storage/reshetnikov/runs/runs/v5/detect/").glob("*")]:    
        path2pred = p_fold / "labels"
        if p_fold.name[:4] == "conf":
            continue
        # path2pred = p_fold / "val"
        names = [Path(x).stem for x in list_ext(path2pred)]
        stat = Statistic("/storage/reshetnikov/openpits/annotations/instances_default.json", path2pred)
        res = stat.test(names)
        print(stat.__str__())
        ans[p_fold]=res['mannwhitneyu']
        ans_ks[p_fold] = res['kstest']

    for k,v in ans.items():
        print(k.name,":",v)
    for k,v in ans_ks.items():
        print(k.name,":",v)
