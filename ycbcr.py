#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:19:59 2021

@author: vijay
"""


import numpy as np
from skimage import feature as skif
import cv2
from PIL import Image



image_path = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/test_preprocess_jan25_1pm/494405.jpg'
def lbp_histogram(image,P=8,R=1,method = 'nri_uniform'):
    '''
    image: shape is N*M 
    '''
    lbp = skif.local_binary_pattern(image, P,R, method) # lbp.shape is equal image.shape
    # cv2.imwrite("lbp.png",lbp)
    max_bins = int(lbp.max() + 1) # max_bins is related P
    hist,_= np.histogram(lbp, density = True, bins=max_bins, range=(0, max_bins))
    return hist
# file_list is a txt file, like this:
# image_path label

def save_features():
    # feature_label = []
    # for line in open(file_list):
        # image_path = line.strip().split(' ')[0]
        # label = int(line.strip().split(' ')[1])
        
    image = Image.open(image_path)
    img = image.convert('YCbCr')
    img = np.array(img)
    y_h = lbp_histogram(img[:,:,0]) # y channel
    cb_h = lbp_histogram(img[:,:,1]) # cb channel
    cr_h = lbp_histogram(img[:,:,2]) # cr channel
    feature = np.concatenate((y_h,cb_h,cr_h))
    print(feature)
        #feature_label.append(np.append(feature,np.array(label)))
    #np.save(file_name,np.array(feature_label))
if __name__ == "__main__":
    save_features()