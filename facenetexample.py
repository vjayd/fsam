#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:02:03 2021

@author: vijay
"""
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms



# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()

from PIL import Image

img_dir = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/test_preprocess_feb1_1pm/494407.jpg'

transform = transforms.Compose([
    transforms.ToTensor()])

    
    
img = Image.open(img_dir)
img = transform(img)
print('') 


# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img.unsqueeze(0))
print(img_embedding)
# Or, if using for VGGFace2 classification
resnet.classify = True
img_probs = resnet(img.unsqueeze(0))