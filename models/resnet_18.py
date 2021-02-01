#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 12:03:40 2021

@author: vijay
"""

import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from facenet_pytorch import MTCNN, InceptionResnetV1

class Resnet_18(nn.Module):
    """
    The class defining Deep Pixel-wise Binary Supervision for Face Presentation Attack
    """

    def __init__(self, pretrained=True):
        super(Resnet_18, self).__init__()
        resnet = models.resnet18(pretrained=pretrained)
        features = list(resnet.children())
        self.backbone = nn.Sequential(*features[0:9])
        
        self.classi = nn.Sequential(
                      nn.Linear(512, 256),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(256, 2))
        
        self.classi = nn.Sequential(
                      nn.Linear(512, 256),  
                      nn.Sigmoid())
       
    def forward(self, x):
        bb = self.backbone(x)
        #print(bb.shape)
        op = self.classi(bb.view(-1, 512))
        
        return op


class Facenet(nn.Module):
    """
    The class defining Deep Pixel-wise Binary Supervision for Face Presentation Attack
    """

    def __init__(self):
        super(Facenet, self).__init__()
        resnet =  InceptionResnetV1(
                    classify=True,
                    pretrained='vggface2'
                    
                    )
        features = list(resnet.children())
        self.convmodel = nn.Sequential(*features[:-3])
        self.classi = nn.Sequential(
                      nn.Linear(1792, 1024),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(1024, 512),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(512, 256),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(256, 1),
                      nn.Sigmoid())
       
        
    def forward(self, x):
        embedding = self.convmodel(x)
        classi = self.classi(embedding.view(-1, 1792))
        return classi



class Resnet_152(nn.Module):
    """
    The class to extract the features trained on Imagenet till the feature vector size is 512
    """

    def __init__(self, pretrained=False):
        super(Resnet_152, self).__init__()
        resnet = models.resnet152(pretrained= pretrained)
        features = list(resnet.children())
        self.backbone = nn.Sequential(*features[:-1])
        
        self.classi = nn.Sequential(
                      nn.Linear(2048, 1024),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(1024, 512),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(512, 256),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(256, 2))
        
        
        
    def forward(self, x):
        bb = self.backbone(x)
        op = self.classi(bb.view(-1, 2048))
        return op
    
    
    
    
    
class Densenet_201(nn.Module):
    
    """
    The class to extract the features trained on Imagenet till the feature vector size is 512
    """

    def __init__(self, pretrained=False):
        super(Resnet_152, self).__init__()
        densenet = models.resnet152(pretrained=pretrained)
        features = list(densenet.children())
        self.backbone = nn.Sequential(*features[:-1])
        self.classi = nn.Sequential(
                      
                      nn.Linear(1920, 1024),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(1024, 512),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(512, 256),  
                      nn.ReLU(), 
                      nn.Dropout(0.3),
                      nn.Linear(256, 2))
        
        
    def forward(self, x):
        
       bb = self.backbone(x)
       op = self.classi(bb.view(-1, 1920))
    
       return op
    
