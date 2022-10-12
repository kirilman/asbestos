import imp
from albumentations import brightness_contrast_adjust, clahe, channel_shuffle, resize, RandomBrightnessContrast
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from dataset import is_image
import sys
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
    # sys.path.append("/home/kirilman/Project/asbestos/yolov5/")

from yolov7.models import Model

def apply_brightness(get_image):
    def wraper(path):
        T = RandomBrightnessContrast()
        return T(image = resize(get_image(path), 1024, 1024), )['image']
    return wraper

@apply_brightness
def read_image(path):
    return cv2.imread(path)

p = Path("/home/kirilman/Project/asbestos/yolov5/runs/detect/exp9")
files = dict()
for path in p.rglob("*"):
    if is_image(path):
        try:
            files[path] = read_image(path.as_posix())
        except Exception as e:
            print("Error in :", path, " ", e)

path_2_model = "/home/kirilman/Project/asbestos/yolov7/runs/train/yolov7_simple/weights/best.pt"
path_2_clf   = "/home/kirilman/Project/asbestos/yolov7/cfg/training/yolov7.yaml"
state = torch.load(path_2_model)
model = Model(cfg=path_2_clf).load_state_dict(state.state_dict())
for k, img in files.items():
    plt.imshow(img)
    break