import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
from PIL import Image
import pandas as pd
import torch

from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import cv2

def plot_masks(segments: List[np.ndarray], fig = None, color = [0,0,1], alpha = 1):
    if fig:
        fig = fig
        ax = fig.gca()
    else:
        fig = plt.figure(figsize = (10,10))
        ax = fig.gca()
        
    for i,label in enumerate(segments):
        polygon = Polygon([(x,y) for x,y in zip(label[1::2],label[2::2])], alpha)
        polygon.set_color(color)
        polygon.set_alpha(alpha)
        ax.add_patch(polygon)
        plt.ylim(0,1024)
        plt.xlim(0,1024)
    return fig

def plot_bboxs(image, bboxs, color = None, line_thickness = None, sline = cv.LINE_AA):
    res_image = image.copy()
    color = color or [255,0,0]
    tl = line_thickness or round(0.002 * (res_image.shape[0] + res_image.shape[1]) / 2) + 1  
    scale_h, scale_w = res_image.shape[:2]
    for bbox in bboxs:
        scale_x = bbox[[0, 2]]*scale_w
        scale_y = bbox[[1, 3]]*scale_h
        c1 = (int(scale_x[0]), int(scale_y[0]))
        c2 = (int(scale_x[1]), int(scale_y[1]))
        res_image = cv.rectangle(res_image, c1, c2, color, tl, lineType=sline)
    return res_image

class Annotator():
    def __init__(self, img):
        self.img = img if isinstance(img, Image.Image) else Image.fromarray(img)
    
    def masks(self, segments, color = [0,0,0], alpha = 1):
        for segment in segments:
            self.img = cv2.fillPoly(np.array(self.img), pts = [segment], color = color)

    def result(self):
        return self.img

        