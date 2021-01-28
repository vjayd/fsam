#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 12:01:30 2018

@author: vijay
"""


import os
import torchvision.datasets as dset
import numpy as np
import random
from PIL import Image
import torch
from os import path
import PIL.ImageOps    
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt


#Class to create a custom datset according to pytorch practices
class Spoof(Dataset):
    
    def __init__(self, imageFolderDataset, dirname, transform=None, should_invert=True, test = False, map_size = 14):
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.directoryname = dirname
        self.should_invert = should_invert
        self.test = test
        self.map_size = 14
        
        
    def random_folder(self, directoryname1):
        folder_name = random.choice(os.listdir(directoryname1))
        return os.path.join(directoryname1, folder_name)
    
    
    
    
        #__get_item is a skeleton provided by the pytorch Dataset we need to override 
    def __getitem__(self,index):
        
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
       
        
        flag = True
        while flag:
            folder = self.random_folder(self.directoryname)
            #if path.exists(folder+'/spoof') and path.exists(folder+'/live'):
                
               
                
            subfolderdataset = dset.ImageFolder(root = folder)
            img0_tuple = random.choice(subfolderdataset.imgs)
            #print(img0_tuple)
            classid = 0
            bbox = img0_tuple[0].split('.j')[0]+'_BB.txt'
            if self.test:
                bbox = img0_tuple[0].split('.p')[0]+'_BB.txt'
            
            #print('print')
            #print(bbox)
            if path.exists(bbox): 
            #print('Length of subfolderataset :',len(subfolderdataset.imgs))
                #print('inside')
               # print(img0_tuple)
                f = open(bbox, "r")
                bbox = f.read().split(' ')
                x1, y1, w, h =  int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                
                if 'spoof' in  img0_tuple[0]:
                    classid = 1
                    mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32)*1
                else:
                    classid = 0
                    mask = np.ones((1, self.map_size, self.map_size), dtype=np.float32)*0
                    
                

                img0 = Image.open(img0_tuple[0])
                newsize = (218, 178) 
                im = img0.resize(newsize)
                img_new = im.crop((x1, y1, x1+w, y1+h))
                img_new = img_new.resize((224,224), resample =3)
                img = self.transform(img_new)
                #print('before break')
                flag = False
                
                #print('returning')
                break
                        #
                        
            
                    
       
        return img, classid, mask, img0_tuple
        
         
         
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)      
    
