import torch
import sys  
sys.path.append('/home/kirilman/Project/asbestos/')
# print(sys.path)
from yolov5.models.yolo import Model
# Create model
device = 'cpu'
im = torch.rand(1, 3, 512, 512).to(device)
model = Model().to(device)
y = model(im)
print(y)