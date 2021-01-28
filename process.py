#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:31:46 2018

@author: vijay
"""
from Config import Config as Config
import matplotlib.pyplot as plt
import torch
from torch import optim
import torch.nn.functional as F
import numpy as np
import gc
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import time



class PixWiseBCELoss(nn.Module):
    """ Custom loss function combining binary classification loss and pixel-wise binary loss
    Args:
        beta (float): weight factor to control weighted sum of two losses
                    beta = 0.5 in the paper implementation
    Returns:
        combined loss
    """
    def __init__(self, beta):
        super().__init__()
        self.criterion = nn.BCELoss()
        self.beta = beta

    
    def forward(self, net_mask, net_label, target_mask, target_label):
        # https://gitlab.idiap.ch/bob/bob.paper.deep_pix_bis_pad.icb2019/blob/master/bob/paper/deep_pix_bis_pad/icb2019/config/cnn_trainer_config/oulu_deep_pixbis.py
        # Target should be the first arguments, otherwise "RuntimeError: the derivative for 'target' is not implemented"
        loss_pixel_map = self.criterion(net_mask, target_mask)
        loss_bce = self.criterion(net_label, target_label)

        loss = self.beta * loss_bce + (1 - self.beta) * loss_pixel_map
        return loss




criterion = PixWiseBCELoss()


class DeepPixBis(nn.Module):
    """
    The class defining Deep Pixel-wise Binary Supervision for Face Presentation Attack
    """

    def __init__(self, pretrained=True):
        super(DeepPixBis, self).__init__()
        dense = models.densenet161(pretrained=pretrained)
        features = list(dense.features.children())
        self.enc = nn.Sequential(*features[0:8])
        self.dec = nn.Conv2d(384, 1, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(14*14, 1)


    def forward(self, x):
        enc = self.enc(x)
        dec = self.dec(enc)
        out_map = F.sigmoid(dec)
        dec = self.linear(out_map.view(-1, 14*14))
        dec = F.sigmoid(dec)
        return out_map, dec
    
    

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


net = DeepPixBis()

def train( train_dataloader, PATH):
#    counter =[]
#    loss_history = []
#    iteration_number = 0
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
    optimizer = optim.Adam(net.parameters(), Config.learning_rate)
    
    if os.path.exists(PATH):
        checkpoint = torch.load(PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_last = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        epoch_last = 0
        loss = 0
    
   
    #inputs, label = next(iter(train_dataloader))
    
    #print('train')
    # Make a grid from batch
    #out = torchvision.utils.make_grid(inputs)

    #imshow(out, title=[x for x in label])
    
    #optimizer = optim.Adam(model_resnet.parameters(), Config.learning_rate)
    
    #criterion = nn.CrossEntropyLoss()
   #print('training')
    #optimizer = optim.SGD(ne.parameters(), lr= 0.001, momentum=0.9)
    for epoch in range(epoch_last, Config.train_epoch):  # loop over the dataset multiple times
        spoof = 0
        total = 0
        running_loss = 0.0
        i = 0
        print('Epoch no : ', epoch)
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #print(i)
            inputs, label, mask, name = data
            #print(inputs.shape,':::::::::', labels)
            inputs = inputs.to(device)
            label = label.to(device)
            mask = mask.to(device)
            
            spoof+=label.sum().item()
            #print(spoof)
            #print(name)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            net_mask, net_label =  net(inputs)
            loss = criterion(net_mask, net_label, mask, label)
            #loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #print(i)
            # print statistics
            running_loss += loss.item()
            total+=1
            #break
        print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / i))
        running_loss = 0.0
        
        print('Total live images ',(total-spoof),' : Total spoof images ',spoof)
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
    
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': running_loss,
                        
                        }, PATH)
    
    
        #torch.save(model_resnet, PATH)
    print('Finished Training')
    
    
    
def test(test_dataloader, PATH):

    '''Visualization of test dataset'''
    inputs, labels, name = next(iter(test_dataloader))

    inputs = inputs.to(device)
    labels = labels.to(device)
    
    
#    out = torchvision.utils.make_grid(inputs)
#    imshow(out, title=[x for x in labels])
    checkpoint = torch.load(PATH)
    model_resnet.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    outputs = model_resnet(inputs.to(device))
    
    _, predicted = torch.max(outputs, 1)
    
#    print('Predicted: ', ' '.join('%5s' % class_names[predicted[j]]
#                                  for j in range(4)))
#    
    correct = 0
    total = 0
    tp , fp, tn, fn = 0, 0, 0, 0
    spoof = 0
    
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            inputs, labels , name= data
            inputs = inputs.to(device)
            labels = labels.to(device)

            spoof+=labels.sum().item()
            
            labels = labels.to(device)
            
            
        #    out = torchvision.utils.make_grid(inputs)
        #    imshow(out, title=[x for x in labels])
            checkpoint = torch.load(PATH)
            model_resnet.load_state_dict(checkpoint['model_state_dict'])
            #optimizer.load_state_dict(checkpoint['op
            outputs = model_resnet(inputs)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            
            ###############################################
            true=(labels==predicted)
            false=(~true)
            pos=(predicted==1)
            neg=(~pos)
            keep=(labels==0)
            tp+=(true*pos).sum().item()
            fp+=(false*pos*keep).sum().item()
            fn+=(false*neg*keep).sum().item()
            tn+=(true*neg).sum().item()
#            print( tp, fp, fn, tn)
#            
#            print(predicted *labels)
#            print(outputs)
#            print(predicted)
#            print(labels)
#            print(correct)
            #break
    print('Total live images ',(total-spoof),' : Total spoof images ',spoof)
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    precision =  tp/(tp+fp) #The proportion of 
    recall =  tp/(tp+fn)
    print('Precision ', precision, 'Recall ', recall)


def analysis(test_dataloader, PATH):
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))
    model_resnet = torch.load(PATH)
    with torch.no_grad():
        for i, data in enumerate(test_dataloader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model_resnet(images)
            
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(2):
                label = i
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(2):
        print('Accuracy of %5s : %2d %%' % (
        i, 100 * class_correct[i] / class_total[i]))







