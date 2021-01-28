#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: vijay
"""

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from Config import Config as Config
from Spoof import Spoof
import process as p
import torch




    
######################################################### TRAIN DATASET ############################################3
folder_dataset = dset.ImageFolder(root = Config.training_dir)
traindataset = Spoof(folder_dataset, Config.training_dir, transform = transforms.Compose([transforms.ToTensor()]), should_invert = False, test= False)
train_dataloader = DataLoader(traindataset, shuffle = True, num_workers = 8, batch_size = Config.batch_size)



########################################################## TEST CODE #############################################


folder_dataset_test = dset.ImageFolder(root=Config.testing_dir)
testdataset = Spoof(folder_dataset_test, Config.testing_dir, transform=transforms.Compose([transforms.ToTensor()]),should_invert=False, test = True)
test_dataloader = DataLoader(testdataset, num_workers=8, batch_size=Config.testbatch_size, shuffle=True)

####################################################### PROCESS THE DATA FOR TRAINING AND TESTING #########################

PATH1 = './without_net.pth'
PATH2 = './with_net.pth'
PATH3 = './withresnet101.pth'
PATH4 = './withresnet101wopretrained101.pth'
PATH5 = './withresnet101.pth'

def process():
     
    p.train(train_dataloader, PATH5)
    p.test(test_dataloader, PATH5)
    #p.analysis(test_dataloader)
   
    
    
    
    
    
    
process()    
