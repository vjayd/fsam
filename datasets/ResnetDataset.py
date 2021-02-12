#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:09:49 2021

@author: vijay
"""

import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage import feature as skif

class ResnetDataset(Dataset):
    """ A data loader for Pixel Wise Deep Supervision PAD where samples are organized in this way

    Args:
        root_dir (string): Root directory path
        csv_file (string): csv file to dataset annotation
        map_size (int): size of pixel-wise binary supervision map. The paper uses map_size=14
        transform: A function/transform that takes in a sample and returns a transformed version
        smoothing (bool): Use label smoothing
    """

    def __init__(self, root_dir, csv_file, map_size, transform=None, smoothing=True):
        super().__init__()
        self.root_dir = root_dir
        self.data = pd.read_csv(os.path.join('', csv_file))
        self.map_size = map_size
        self.transform = transform
        
    def lbp_histogram(self, image,P=8,R=1,method = 'nri_uniform'):
        '''
        image: shape is N*M 
        '''
        lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
        # cv2.imwrite("lbp.png",lbp)
        max_bins = int(lbp.max() + 1) # max_bins is related P
        hist,_= np.histogram(lbp, density=True, bins=max_bins, range=(0, max_bins))
        return hist


    def __getitem__(self, index):
        """ Get image, output map and label for a given index
        Args:
            index (int): index of image
        Returns:
            img (PIL Image): 
            mask: output map (14x14)
            label: 1 (genuine), 0 (fake) 
        """
        img_name = self.data.iloc[index, 0]
        img_name = os.path.join(self.root_dir, img_name)
        img = Image.open(img_name)
        
        image = img.convert('YCbCr')
        image = np.array(image)
        y_h = self.lbp_histogram(image[:,:,0]) # y channel
        cb_h = self.lbp_histogram(image[:,:,1]) # cb channel
        cr_h = self.lbp_histogram(image[:,:,2]) # cr channel
        # print(y_h.shape)
        # print(cb_h.shape)
        # print(cr_h.shape)
        
        if (y_h.shape == cb_h.shape == cr_h.shape) :
            feature = np.concatenate((y_h,cb_h,cr_h)).astype(np.float32)
        else:
            feature = np.ones((177), dtype=np.float32) 
        label = self.data.iloc[index, 1].astype(np.float32)
        label = np.expand_dims(label, axis=0)
        if label == 1:
            mask = np.ones((self.map_size), dtype=np.float32) 
        else:
            mask = np.zeros((self.map_size), dtype=np.float32) 
        
        if self.transform:
            img = self.transform(img)

        return img, label , mask, feature


    def __len__(self):
        return len(self.data)
