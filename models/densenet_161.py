import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F

class DeepPixBis(nn.Module):
    """
    The class defining Deep Pixel-wise Binary Supervision for Face Presentation Attack
    """

    def __init__(self, pretrained=True):
        super(DeepPixBis, self).__init__()
        dense = models.densenet161(pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[0:6])
        self.dec = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)
        #self.linear = nn.Linear(14*14, 1)  # the actual densenet code
        self.classi = nn.Sequential(
                      nn.Linear(28*28, 192),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(192, 2))


    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        #dec = self.linear(out_map.view(-1, 14*14)) #the actual densenet code
        dec = self.classi(out_map.view(-1, 28*28))
        dec = F.sigmoid(dec)
        return out_map, dec
