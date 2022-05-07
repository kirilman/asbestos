import imp
from turtle import forward
from torch import nn
from .unet import Unet, Attention_Unet
import torch.nn.functional as F

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
    
class DiceLoss_and_CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.entropy = nn.CrossEntropyLoss()
    def forward(self, inputs, targets):
        dice_value = self.dice.forward(inputs, targets)
        entropy_value = self.entropy.forward(inputs, targets)
        return dice_value + entropy_value


ALPHA = 0.5
BETA = 0.5
"https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/dice_loss.py"

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky