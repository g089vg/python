# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 09:03:03 2018

@author: g089v
"""

import os
import cv2
import numpy
from PIL import Image
import six

import numpy as np

from io import BytesIO
import os
import pickle
import json
import numpy as np

import skimage.io as io

from chainer.dataset import dataset_mixin

# download `BASE` dataset from http://cmp.felk.cvut.cz/~tylecr1/facade/
class CrackDataset(dataset_mixin.DatasetMixin):
    def __init__(self, dataDir='./crack/', data_range=(1,88)):
        print("load dataset start")
        self.dataDir = dataDir
        self.dataset = []

        data_dir_path1 = u"./crack_image/"
        data_dir_path2 = u"./crack_label/"
        nnum = 0
        file_list = os.listdir(r"./crack_image/")
        for file_name in file_list:
            root, ext = os.path.splitext(file_name)
            nnum = nnum + 1
            if ext == u'.jpg' and nnum >=data_range[0] and  nnum <=data_range[1]:
                abs_name1 = data_dir_path1  + file_name
                abs_name2 = data_dir_path2  + file_name[:-4]+".png"
#                print(abs_name1)
#                print(abs_name2)
        #        img = cv2.imread(abs_name1)
        #        label = cv2.imread(abs_name2)
                img = Image.open(abs_name1)
                img = img.convert("RGB")
                label = Image.open(abs_name2)
                label = label.convert("P")
                w,h = img.size
        
                r = 350 / float(min(w,h))
                img = img.resize((int(r*w), int(r*h)), Image.BILINEAR)
                label = label.resize((int(r*w), int(r*h)), Image.NEAREST)
        
                img = np.asarray(img).astype("f").transpose(2,0,1)/128.0-1.0
                label_ = np.asarray(label)-1 # [0, 12) 
                label2 = np.zeros((5, img.shape[1], img.shape[2])).astype("i")
        #                print(label.shape)
        #                print(label_.shape)
                for j in range(5):
                        label2[j,:] = label_==j
                self.dataset.append((img,label2))                  


        print("load dataset done")

    def __len__(self):
        return len(self.dataset)

    # return (label, img)
    def get_example(self, i, crop_width=256):
                
        _,h,w = self.dataset[i][0].shape
        x_l = np.random.randint(0,w-crop_width)
        x_r = x_l+crop_width
        y_l = np.random.randint(0,h-crop_width)
        y_r = y_l+crop_width
        return self.dataset[i][1][:,y_l:y_r,x_l:x_r], self.dataset[i][0][:,y_l:y_r,x_l:x_r]

    
    
    
#dataset = []    
#data_dir_path1 = u"./crack_image/"
#data_dir_path2 = u"./crack_label/"
#
#file_list = os.listdir(r"./crack_image/")
#for file_name in file_list:
#    root, ext = os.path.splitext(file_name)
#    if ext == u'.jpg':
#        abs_name1 = data_dir_path1  + file_name
#        abs_name2 = data_dir_path2  + file_name[:-4]+".png"
#        
#        img = cv2.imread(abs_name1)
#        label = cv2.imread(abs_name2)
#        dataset.append((img,label))
#        
#        img = np.asarray(np.float32(img)/255.0)
#        label_ = np.asarray(label)-1  # [0, 12)
##                print(label)
#        label = np.zeros((2, img.shape[0], img.shape[1])).astype("i")
##        
#        for j in range(2):
#            for y in range(50):
#                for x in range(50):
#                    if j == 0:
#                        label[j][y][x] = label_[y][x][0] 
#                    if j == 1:
#                        label[j][y][x] = label_[y][x][2]             
            
            
            
            
