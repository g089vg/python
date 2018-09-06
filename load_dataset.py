# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:54:18 2018

@author: g089v
"""
from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def Load_Dataset(label0="./inflation_0/",label1="./inflation_1/",channels = 1):
#    number = 45552*2
    pathsAndLabels = []
    pathsAndLabels.append(np.asarray([label0, 0]))
    pathsAndLabels.append(np.asarray([label1, 1]))
    
    # データを混ぜて、trainとtestがちゃんとまばらになるように。
    allData = []
    for pathAndLabel in pathsAndLabels:
        path = pathAndLabel[0]
        label = pathAndLabel[1]
        imagelist = glob.glob(path + "*")
        for imgName in imagelist:
            allData.append([imgName, label])
    allData = np.random.permutation(allData)
    imageData = []
    labelData = []

    if channels == 1: 
        for pathAndLabel in allData:
    #        print(pathAndLabel[0])
            img = np.array(Image.open(pathAndLabel[0]).convert('L') )
    
            grayImgData = np.asarray(np.float32(img)/255.0)
            imgData = np.asarray([grayImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))
    
    if channels == 3:

        for pathAndLabel in allData:
            img = Image.open(pathAndLabel[0])
            #3チャンネルの画像をr,g,bそれぞれの画像に分ける
            r,g,b = img.split()
            rImgData = np.asarray(np.float32(r)/255.0)
            gImgData = np.asarray(np.float32(g)/255.0)
            bImgData = np.asarray(np.float32(b)/255.0)
            imgData = np.asarray([rImgData, gImgData, bImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))
            
            
    threshold = np.int32(len(imageData)/8*7)
    train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
    test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
    print("train={0}".format(threshold))
#    print("test={0}".format(number-threshold))
    return train,test