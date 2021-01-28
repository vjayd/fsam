#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 12:12:08 2021

@author: vijay
"""

import csv
import os
live = 0
spoof = 0
with open('./trainpre_jan12_11pm.csv', 'w', newline='') as file:
    with open('./trainpre_jan6_8pm.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        for row in csv_reader:
            classid = int(row[1])
            #print(classid)
            if classid == 1 and spoof <=25000:
                writer = csv.writer(file)
                writer.writerow(row)
                spoof+=1
                
            elif classid == 0 and live <=50000:
                writer = csv.writer(file)
                writer.writerow(row)
                live+=1
            print('Spoof : ',spoof, ': Live :', live)