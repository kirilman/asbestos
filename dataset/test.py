import numpy as np
from PIL import Image
import os 
from pathlib import Path
root = "/home/kirill/Учеба/asbestos/task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/images/asbestos/stones/lab_common_camera/"
p = "/home/kirill/Учеба/asbestos/task_asbestos_stone_lab_common_camera-2021_12_10_13_12_14-mots png 1.0/images/asbestos/stones/lab_common_camera/IMG_2688.png"
files = os.listdir(root)
img = np.array(Image.open(p).convert('L'), dtype = np.float32)/255
print(img)
print(img.shape)

c = 5
images = {}
for i, f in enumerate(files):
    images[i] = np.array(Image.open(Path(root,f)).convert('L'), dtype = np.float32)/255
    if i > c:
        break

print(images[2])