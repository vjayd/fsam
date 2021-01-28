#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:07:14 2021

@author: vijay
"""
'''
File to create a feature extraction npy file from the resnet_18 model

'''


import torch
from torchvision import transforms
from torch import nn
from torchvision import models
import torch.nn.functional as F
import csv
import os
from PIL import Image
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score
from pyod.models.copod import COPOD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import statistics
#data_dir = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/train_preprocess_8pm'
#csv_dir = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/trainpre_jan6_8pm.csv'
npy_name_train = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/train_resnet18_preprocess.npy'
data_dir = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/test_preprocess_jan6_8pm'
csv_dir = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/testpre_jan6_8pm.csv'
npy_name_test = '/media/vijay/1TB/FACE_ANTI-SPOOFING/antispoof_data/final_data/CelebA_Spoof/Data/test_resnet18_preprocess.npy'


device  = torch.device("cuda:0" if torch.cuda.is_available() else "cuda:0")
print(device)

class Resnet_18_512(nn.Module):
    """
    The class to extract the features trained on Imagenet till the feature vector size is 512
    """

    def __init__(self, pretrained=True):
        super(Resnet_18_512, self).__init__()
        resnet = models.resnet18(pretrained=True)
        features = list(resnet.children())
        self.backbone = nn.Sequential(*features[0:9])
        
    def forward(self, x):
        bb = self.backbone(x)
       
        
        return bb.view(-1, 512)
    
#np.save(file_name1, feature_label)


class Resnet_152_2048(nn.Module):
    """
    The class to extract the features trained on Imagenet till the feature vector size is 512
    """

    def __init__(self, pretrained=True):
        super(Resnet_152_2048, self).__init__()
        resnet = models.resnet152(pretrained=True)
        features = list(resnet.children())
        self.backbone = nn.Sequential(*features[:-1])
        
    def forward(self, x):
        bb = self.backbone(x)
       
        
        return bb.view(-1, 2048)



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
    
    
    
    
    
net = DeepPixBis().to(device)

def create_features(csv_dir, data_dir, npy_name):
    with open(csv_dir) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter = ',')
        feat = []
        cnt = 0
        for row in csv_reader:
            classid = int(row[1])
            img_name = row[0].split(".")[0]+'.jpg'
            #img_name = row[0]
            img_path = os.path.join(data_dir, img_name)
           #print(os.path.exists(img_path))
            if os.path.exists(img_path):
                img_pil = Image.open(img_path)
                img_pil_tensor = transforms.ToTensor()(img_pil).unsqueeze_(0).to(device)
                op = net(img_pil_tensor).to(device) 
                #op1= []
                op1 = np.asarray(op.detach().cpu().numpy())[0].tolist()
                op1.append(classid)
                
                feat.append(op1)
                cnt+=1
                print(cnt)
                
    #            
            
               # if cnt==5000:
      #  feat1 = np.asarray(feat)
        #np.save(npy_name, feat1 )
                #    break
    #            X = feat1[:,0:511]
    #            Y = feat1[:,512]
    #            # fit the model
    #            clf = svm.NuSVC(gamma='auto')
    #            clf.fit(X, Y)
               
create_features(csv_dir, data_dir, npy_name_test)


def remove_outlier(data, x, row, contamination):
    '''
     data: which kind of data you are passing
     x : 0 for live and 1 for spoof
    '''
    
    # 0 indicates all the live images
    x1 = np.where(data[:, 512]==x) #512
    x2 = data[x1][0:row, :]
    train_features_x = x2[:,0:511]
    clf = COPOD(contamination = contamination)
    clf.fit(train_features_x)
    z = clf.labels_
    z = np.asarray(z).reshape(row,1)
    z_final =  np.hstack((x2, z))
    
    x1_2 = np.where(z_final[:, 513]==0)
    x2_2 = z_final[x1_2][:, :]
    return x2_2




def mean_std(data, row, x, test = False):
    
   
    x1 = np.where(data[:, 512]==x) #512
    x2 = data[x1][0:row, :]
    train_features_x = x2[:, 0:511]
    #train_features_x = StandardScaler().fit_transform(train_features_x)
    
    tm = train_features_x.mean(axis = 0)
    std = train_features_x.std(axis = 0)
    ads = train_features_x - tm
    
    #for j in np.arange(0.00, 50.00, 0.5):
    if test:
        ads1 = abs(ads)<=2
    else:
        ads1 = abs(ads)<=0.12
    zed = np.all(ads1, axis =1)
    print(zed.sum())
    z = np.asarray(zed).reshape(row,1)
    z_final =  np.hstack((x2, z))
    
    x1_2 = np.where(z_final[:, 513]==1)
    x2_2 = z_final[x1_2][:, :]
    return x2_2
    
    
    

def train_test(npy_train, npy_test):
    '''
    
    '''
    
    
    
    train = np.load(npy_train)
    test = np.load(npy_test)
    tot_rows = 50000
    
    #data =  mean_std(train, tot_rows, 0)
    #sd, mean = mean_std(train, tot_rows, 1)
#    print(sd, ' and mean is : ', mean)
#    print(sd, ' and mean of spoof is : ', mean)
    
    
    
    
    
    live_train = mean_std(train, tot_rows, 0, test = False)
    spoof_train = mean_std(train, tot_rows, 1, test = False)
#    live_train = remove_outlier(train, 0, tot_rows, 0.1)
#    spoof_train = remove_outlier(train, 1, tot_rows, 0.5)
    train = np.vstack((live_train, spoof_train))
    print('Train sample size :', train.shape)
    train_x = train[:, 0:511]
    train_y = train[:,512]
    
    
    tot_rows_test = 11000
    live_test = mean_std(test, tot_rows_test, 0, test = True)
    spoof_test = mean_std(test, tot_rows_test, 1, test = True)
#    live_test = remove_outlier(test, 0, tot_rows_test, 0.1)
#    spoof_test = remove_outlier(test, 1, 27000, 0.2)
    test = np.vstack((live_test, spoof_test))
    print('Test sample size :', test.shape)
    test_x = test[:, 0:511]
    test_y = test[:,512]
    
    
    
    
    
    
    
    
    
    for i in range(3, 500, 5):
    # fit the model
        print(i)
        clf = svm.SVC( kernel = 'poly', degree = i, gamma ='auto', cache_size =20000, verbose = False,  max_iter= 1000) #check iterations
        #clf = RandomForestClassifier(max_depth=100, random_state=0, n_estimators = 1000)
        #train_x = StandardScaler().fit_transform(train_x)
        clf.fit(train_x, train_y)
        print('fitting done')
        #test_x = StandardScaler().fit_transform(test_x)
        predictions_poly = clf.predict(test_x)
        #print('predictions done for :', i)
        print(predictions_poly.sum())
        accuracy_poly = accuracy_score(test_y, predictions_poly)
        print(accuracy_poly)
        #acc[i] = accuracy_poly
        #return clf
       # print(acc)
              

#train_test(npy_name_train, npy_name_test)      



