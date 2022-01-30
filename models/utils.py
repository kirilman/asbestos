import imp
from torch import nn
from .unet import Unet, Attention_Unet

def get_network(name):
    if name == 'Unet':
        return  Unet
    if name == 'Attention_Unet':
        return Attention_Unet
    else:
        raise "Model not found"

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
    def forward(self, inputs, targets, smooth=1e-5):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice
    
