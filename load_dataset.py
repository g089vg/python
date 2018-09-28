# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:54:18 2018

@author: g089v
"""
from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob
import os
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
def Load_Dataset(Folder_Name, mode,channels):

    allData = []
    file_list = os.listdir(Folder_Name) 
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            file_name =  file_name[:-4]

            label0 ="./data/Ground_Truth/"+file_name+"/0/inf_"+mode+"/"
            label1 = "./data/Ground_Truth/"+file_name+"/1/inf_"+mode+"/"
    
            pathsAndLabels = []
            pathsAndLabels.append(np.asarray([label0, 0]))
            pathsAndLabels.append(np.asarray([label1, 1]))
    
            # データを混ぜて、trainとtestがちゃんとまばらになるように。
    
            for pathAndLabel in pathsAndLabels:
                
                path = pathAndLabel[0]
                label = pathAndLabel[1]
                imagelist = glob.glob(path + "*")
                for imgName in imagelist:
                    allData.append([imgName, label])
            
    allData = random.sample(allData, len(allData))
    number = len(allData)   
    number = float(number)
    print("number",number)
    load_num = 0
    load_rate = 0.0

    if channels == 1: #チャンネル数が1つの場合
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
#            print(pathAndLabel[0])
            img = np.array(Image.open(pathAndLabel[0]).convert('L') )
    
            grayImgData = np.asarray(np.float32(img)/255.0)
            imgData = np.asarray([grayImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))
        f_image = open('image.txt', 'w')         
        list = imageData
        pickle.dump(list, f_image)
        f_label = open('label.txt', 'w')
        list = labelData
        pickle.dump(list, f_label)            
        threshold = np.int32(len(imageData)/8*7)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        print("train={0}".format(threshold))
        print("test={0}".format(number-threshold))
        return train,test    
            
    if channels == 3:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
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
        print("test={0}".format(number-threshold))
        return train,test 
           
    
    if channels == 31:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
            img11 = np.array(Image.open(pathAndLabel[0]).convert('L') )
    
            
            if int(pathAndLabel[1]) == 0 :
                path31 =pathAndLabel[0].replace('median11', 'median21')
                path51 =pathAndLabel[0].replace('median11', 'median41')
            if int(pathAndLabel[1]) == 1 :
                path31 =pathAndLabel[0].replace('median11', 'median21')
                path51 =pathAndLabel[0].replace('median11', 'median41')
                
    #            print(path)
            img31 =  np.array(Image.open(path31).convert('L') )
            img51 =  np.array(Image.open(path51).convert('L') )
            #3チャンネルの画像をr,g,bそれぞれの画像に分ける     
            grayImgData11 = np.asarray(np.float32(img11)/255.0)
            grayImgData31 = np.asarray(np.float32(img31)/255.0)
            grayImgData51 = np.asarray(np.float32(img51)/255.0)
            imgData = np.asarray([grayImgData11,grayImgData31,grayImgData51])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))        
        threshold = np.int32(len(imageData)/8*7)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])        
    
        print("train={0}".format(threshold))
        print("test={0}".format(number-threshold))
        return train,test   
        
        
    if channels == 4:
    
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
            img = Image.open(pathAndLabel[0])
            path = pathAndLabel[0][len(label0):]
            
            if int(pathAndLabel[1]) == 0 :
                path =pathAndLabel[0].replace('rgb', 'median41')
            if int(pathAndLabel[1]) == 1 :
                path =pathAndLabel[0].replace('rgb', 'median41')
    #            print(path)
            img2 = np.array(Image.open(path).convert('L') )
    
            #3チャンネルの画像をr,g,bそれぞれの画像に分ける
            r,g,b = img.split()
            rImgData = np.asarray(np.float32(r)/255.0)
            gImgData = np.asarray(np.float32(g)/255.0)
            bImgData = np.asarray(np.float32(b)/255.0)       
            grayImgData = np.asarray(np.float32(img2)/255.0)
            imgData = np.asarray([rImgData, gImgData, bImgData,grayImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))        
        threshold = np.int32(len(imageData)/1000*999)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])        
        
        print("train={0}".format(threshold))
        print("test={0}".format(number-threshold))
        return train,test
    
