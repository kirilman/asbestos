import cv2 
import numpy as np

class Image():
    def __init__(self, path):
        self._object = cv2.read(path)
    
    def data(self):
        return
    
    def shape(self):
        return self._object.shape

    def name(self):
        return self._name

