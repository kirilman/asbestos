from re import S
from cffi.model import qualify
import numpy as np
import cv2 as cv
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from typing import List, Tuple, Union


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


def drawline(img,pt1,pt2,color,thickness=1,gap=10):
    dist =((pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2)**.5
    pts= []
    for i in  np.arange(0,dist,gap):
        r=i/dist
        x=int((pt1[0]*(1-r)+pt2[0]*r)+.5)
        y=int((pt1[1]*(1-r)+pt2[1]*r)+.5)
        p = (x,y)
        pts.append(p)
    s=pts[0]
    e=pts[0]
    i=0
    for p in pts:
        s=e
        e=p
        if i%2==1:
            cv.line(img,s,e,color,thickness)
        i+=1

def drawpoly(img,pts,color,thickness):
    s=pts[0]
    e=pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s=e
        e=p
        drawline(img,s,e,color,thickness)

def drawrect(img,pt1,pt2,color,thickness=1):
    pts = [pt1,(pt2[0],pt1[1]),pt2,(pt1[0],pt2[1])] 
    drawpoly(img,pts,color,thickness)

def read_img(path):
    image = Image.read(path)

class Annotator():
    def __init__(self, img, line_width = None):
        self.img = img if isinstance(img, Image.Image) else Image.fromarray(img)
        self.draw = ImageDraw.Draw(self.img)
        h,w = self.img.size
        self.lw = line_width or max(round(sum([h,w]) / 2 * 0.003), 2)  # line width
        print(self.img.size)

    def masks(self, segments:List[np.array], color = [0,0,0], alpha = 0.9):
        """
            List segments: np.array([[x1, y1], [x2, y2], .. ,[xn,yn]])
        """
        assert isinstance(segments, List) and type(segments[0][0][0] == np.int64), 'segments not List'
        for segment in segments:
            assert len(segment.shape) == 2, "Every segment have shape size 2"
        image = self.img.copy()
        for segment in segments:
            image = cv.fillPoly(np.array(image), pts = [segment], color = color) 
        
        image = cv.addWeighted(np.array(self.img)   , alpha, image, 1 - alpha, 0.0)
        #Копируем обратно
        self.img = Image.fromarray(image)
        self.draw = ImageDraw.Draw(self.img)


    def result(self):
        return self.img

    def add_box(self, 
                box: List, 
                label: str = '', 
                color: Union[Tuple,List] = (128,128,128),
                style = None,
                thickness = 3):
        """
            box: [x1, y1, x2, y2]
            Add bbox on image
        """
        if isinstance(box, np.ndarray):
            box = box.tolist()
        if len(np.array(self.img).shape) == 1:
            color = np.mean(color)
        if style == "dotted":
            image = np.array(self.img)
            x1, y1, x2, y2 = box
            drawrect(image, (x1, y1), (x2, y2), color, thickness)
            self.img = Image.fromarray(image)
            self.draw = ImageDraw.Draw(self.img, width = thickness)
        else:
            self.draw.rectangle(box, width = self.lw, outline = color)

    def add_polygone(self, polygone: List[Tuple[float, float]], color = (128, 128, 0)):
        # [Tuple()]
        self.draw.polygon(polygone, width=self.lw, outline = color)
        

    def add_text(self, x,y, text, color= (0,256,0), thickness = 2):
        image = np.array(self.img)
        image = cv.putText(image, str(text), (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, color, thickness, cv.LINE_AA)   
        self.img = Image.fromarray(image)

    def save(self, filename):
        self.img.save(filename, quality = 100)

    @property
    def image(self):
        return np.array(self.img.convert("RGB"))
        
if __name__ == '__main__':
    img = cv.imread('/storage/reshetnikov/open_pits_merge/images/120.bmp')
    ann = Annotator(img,25)
    # ann.add_box([5,5, 1000, 1000])
    t = np.linspace(0, 2*np.pi, 180)
    R = 200.
    xx = 1000 + R*np.cos(t)
    yy = 1000 + R*np.sin(t)
    polygone = np.zeros(len(t)*2)
    polygone = [(x,y) for x,y in zip(xx,yy)]
    ann.add_polygone(polygone, color=(128,0,128))
    img = ann.image
    print(img.shape, ann.img.size)
    plt.imshow(img)
    plt.show()
    ann.img.save('Test.jpeg')
