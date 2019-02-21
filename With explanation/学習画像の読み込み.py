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
import os.path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from random import random

#画像の読み込みに関する関数
#各チャンネル数ごとに場合分け
def Load_Dataset(Folder_Name, mode,channels, num):


    allData = []

#   フォルダから画像を取得 
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
    

            for pathAndLabel in pathsAndLabels:
                
                path = pathAndLabel[0]
                label = pathAndLabel[1]
                imagelist = glob.glob(path + "*")
                for imgName in imagelist:
                    allData.append([imgName, label])
                    
# データを混ぜて、trainとtestがちゃんとまばらになるように。            
    allData = np.random.permutation(allData)

    number = len(allData)  
    print("number",number)
    
    
    load_num = 0
    load_rate = 0.0


    if channels == 1: #チャンネル数が1つの場合（メディアン補正）
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

            
        threshold = len(imageData)*0.9 
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        np.save('./np_data/image_95-5_'+num+mode+'.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'.npy', labelData)
        print(len(imageData))
        print(threshold)
        return train,test    
  
    if channels == 13: #メディアン補正と領域分割結果を合わせて入力
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
#            print(pathAndLabel[0])
            img = np.array(Image.open(pathAndLabel[0]).convert('L') )
            
#           領域分割結果を読み込み（同画像名）存在しない場合を黒の画像
            if os.path.isfile("./data/seg_hall_inf/"+pathAndLabel[0]) == True:
                seg_img1 = np.array(Image.open("./data/seg_hall_inf/"+pathAndLabel[0]))
            else :
                seg_img1 = np.array(np.full((0,0,), 0, dtype=np.uint8))

            if os.path.isfile("./data/seg_shadow_inf/"+pathAndLabel[0]) == True:
                seg_img2 = np.array(Image.open("./data/seg_shadow_inf/"+pathAndLabel[0]))
            else :
                seg_img2 = np.array(np.full((0,0,), 0, dtype=np.uint8))
                
            if os.path.isfile("./data/seg_hyouzi_inf/"+pathAndLabel[0]) == True:
                seg_img3 = np.array(Image.open("./data/seg_hyouzi_inf/"+pathAndLabel[0]))
            else :
                seg_img3 = np.array(np.full((0,0,), 0, dtype=np.uint8))
                

    
            grayImgData = np.asarray(np.float32(img)/255.0)
            seg1  = np.asarray(np.float32(seg_img1)/255.0)
            seg2  = np.asarray(np.float32(seg_img2)/255.0)
            seg3  = np.asarray(np.float32(seg_img3)/255.0)
            
            
            imgData = np.asarray([grayImgData,seg1,seg2,seg3])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))

            
        threshold = len(imageData)*0.9 
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        np.save('./np_data/image_95-5_'+num+mode+'.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'.npy', labelData)
        print(len(imageData))
        print(threshold)
        return train,test    
          
    if channels == 3: # RGBカラーの場合
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

        threshold = len(imageData) -100 
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        np.save('./np_data/image_95-5_'+num+mode+'.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'.npy', labelData)
        print(len(imageData))
        print(threshold)
        return train,test 
           
    
