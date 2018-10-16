# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:55:20 2018

@author: g089v
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:41:29 2018

@author: g089v
"""

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from chainer import serializers
from simple_convnet3 import CNN
import chainer.links as L
import cv2
from PIL import Image
from F1measure import F1_measure
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F

if __name__ == '__main__':

    filter_num = 11
    filter_str = str(filter_num) 

            
    model =  L.Classifier(CNN())
#    serializers.load_npz("../holdout/5/train_rgb/CNN_model.npz", model)

    serializers.load_npz("./zmodels/CNN_modelrgb100.npz", model)
    
    val_num = 0.6

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    data_channels = 1
    data_dir_path1 = u"./data/median41"
    data_dir_path2 = u"./data/half_data"
    file_list = os.listdir(r'../holdout/1/1_test/')
            
    nnum = 0
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            nnum = nnum + 1
            print(file_name)
            abs_name1 = data_dir_path1 + '/' + file_name
            abs_name2 = data_dir_path2 + '/' + file_name
            file_name = file_name[:-4]
            
            if data_channels == 3:
                src_img = cv2.imread(abs_name1)
                height, width,channela = src_img.shape
            
            if data_channels == 1:
                src_img = cv2.imread(abs_name1,0)
                height, width = src_img.shape    
                
            if data_channels == 31:
                src_img = cv2.imread(abs_name1,0)
#                print("./data/median31/"+file_name)
#                print(abs_name1)
                src_img31 = cv2.imread("./data/nlm31/"+file_name+".bmp",0)
                src_img51 = cv2.imread("./data/nlm51/"+file_name+".bmp",0)
                height, width = src_img.shape   
#                cv2.imshow("a",src_img)
#                cv2.imshow("b",src_img31)
#                cv2.imshow("c",src_img51)
#                cv2.waitKey(0)                
                
            dst_img = cv2.imread(abs_name2)            
            f1_img = cv2.imread(abs_name2)
           
            
#           

            

            
            mask  = np.zeros((height, width), np.uint8)
#            cv2.imshow("a",mask)
#            cv2.waitKey(0)            
            height_split = 5
            width_split = 7
            new_img_height = int(height / height_split)
            new_img_width = int(width / width_split)
           
            num  = 0
            for h in range(height_split):
                height_start = h * new_img_height
                height_end = height_start + new_img_height
        
                for w in range(width_split):
                    
                    width_start = w * new_img_width
                    width_end = width_start + new_img_width
                    num = num +1

                    clp1 = src_img[height_start:height_end, width_start:width_end]
                    PIL_data=Image.fromarray(clp1)


                    if data_channels == 3:
                        
                        r,g,b = PIL_data.split()
                        rImgData = np.asarray(np.float32(r)/255.0)
                        gImgData = np.asarray(np.float32(g)/255.0)
                        bImgData = np.asarray(np.float32(b)/255.0)
                        imgData = np.asarray([rImgData, gImgData, bImgData])
    #                    grayImgData = np.asarray(np.float32(PIL_data)/255.0)
    
                        x = imgData
                    
                    if data_channels == 1:
                        grayImgData = np.asarray(np.float32(PIL_data)/255.0)
                        x = grayImgData[None,...]
                    
                    if data_channels == 31:
                        clp31 = src_img31[height_start:height_end, width_start:width_end]
                        PIL_data31=Image.fromarray(clp31)
                        
                        clp51 = src_img51[height_start:height_end, width_start:width_end]
                        PIL_data51=Image.fromarray(clp51)
                        
                        grayImgData11 = np.asarray(np.float32(PIL_data)/255.0)
                        grayImgData31 = np.asarray(np.float32(PIL_data31)/255.0)
                        grayImgData51 = np.asarray(np.float32(PIL_data51)/255.0)                        
                        imgData = np.asarray([grayImgData11, grayImgData31, grayImgData51])
                        x = imgData
                    
                    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                        y = model.predictor(x[None,...]).data.argmax(axis=1)[0]
                        yy = model.predictor(x[None,...])
                        rate = F.softmax(yy.data)[0][1]

#                        print(rate.data*100)
#                    if rate.data >= 0.4:
                    if y == 1:
#                        print('CNN:  crack:')
#                        plt.imshow(grayImgData, cmap='gray')
#                        plt.show()
                        for y  in range(height_start,height_end):
                            for x  in range(width_start,width_end):
                               mask[y][x] = 255
                               dst_img[y][x][2] = 255
            a,b,c,d = F1_measure(f1_img,mask,file_name)
            TP = TP + a
            FP = FP + b
            FN = FN + c
            TN = TN + d

        
            cv2.imwrite('CNN_output/'+file_name+'.bmp', dst_img)     
    Precision = (TP+0.001)/(TP+FP+0.001)
    Recall = (TP+0.001)/(TP+FN+0.001)
    F1 = 2*Recall*Precision/(Recall+Precision)
    Specificity = (TN+0.001)/(TN+FP+0.001)      
    
    print ("Precision={:.4}".format(Precision))
    print ("Recall={:.4}".format(Recall))
    print ("Specificity={:.4}".format(Specificity)) 
    print ("F1={:.4}\n\n".format(F1))
                    

    

