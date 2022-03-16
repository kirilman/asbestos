import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os
import sys
from PIL import Image
import pandas as pd
import torch

def plot_bboxs(image, bboxs, color = None, line_thickness = None ):
    color = color or [255,0,0]
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  
    scale_h, scale_w = image.shape[:2]
    for bbox in bboxs:
        scale_x = bbox[[0, 2]]*scale_w
        scale_y = bbox[[1, 3]]*scale_h
        c1 = (int(scale_x[0]), int(scale_y[0]))
        c2 = (int(scale_x[1]), int(scale_y[1]))
        image = cv.rectangle(image, c1, c2, color, tl, lineType=cv.LINE_AA)
    return image