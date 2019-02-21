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
#ネットワーク構造に対応
from simple_convnet4 import CNN

import chainer.links as L
import cv2
from PIL import Image
from F1measure import F1_measure
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F

def predict(filter_num = 5,save_file = ""):


    filter_str = str(filter_num) 

#    モデルの定義
    model =  L.Classifier(CNN())
#   モデルの読み込み
    serializers.load_npz("./epoch-50", model, path= 'updater/model:main/')     


    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    data_channels = 13
    data_dir_path1 = u"./data/2.5m_median41"
    data_dir_path2 = u"./data/2.5m_half"
    file_list = os.listdir(r'./data/2.5m_half/')

    nnum = 0
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        nnum = nnum + 1
        if ext == u'.bmp' :
            
            print(file_name,nnum)
            abs_name1 = data_dir_path1 + '/' + file_name
            abs_name2 = data_dir_path2 + '/' + file_name
            file_name = file_name[:-4]
            
#            使用する画像によって読み込み
            if data_channels == 3 or data_channels == 33 :
                src_img = cv2.imread(abs_name1)
                height, width,channela = src_img.shape
            
            if data_channels == 1 or data_channels == 13:
                src_img = cv2.imread(abs_name1,0)
                height, width = src_img.shape    
              
                
            dst_img = cv2.imread(abs_name2) 
            gray_img = cv2.imread(abs_name2,0) 
            f1_img = cv2.imread(abs_name2)
           

            mask  = np.zeros((height, width), np.uint8)
         
#            画像の分割設定縦５，横７に分割
            height_split = 5
            width_split = 7
            new_img_height = int(height / height_split)
            new_img_width = int(width / width_split)
            a1,b1,c1 = 0,0,0           
            num  = 0
            for h in range(height_split):
                height_start = h * new_img_height
                height_end = height_start + new_img_height
        
                for w in range(width_split):
                    
                    width_start = w * new_img_width
                    width_end = width_start + new_img_width
                    num = num +1
                    
#                    分割画像
                    clp1 = src_img[height_start:height_end, width_start:width_end]
                    PIL_data=Image.fromarray(clp1)

#                   RGBカラー画像
                    if data_channels == 3:
                        
                        r,g,b = PIL_data.split()
                        rImgData = np.asarray(np.float32(r)/255.0)
                        gImgData = np.asarray(np.float32(g)/255.0)
                        bImgData = np.asarray(np.float32(b)/255.0)
                        imgData = np.asarray([rImgData, gImgData, bImgData])
    
                        x = imgData

#                   RGBカラー画像と領域分割結果
                    if data_channels == 33:
                        
                        r,g,b = PIL_data.split()
                        rImgData = np.asarray(np.float32(r)/255.0)
                        gImgData = np.asarray(np.float32(g)/255.0)
                        bImgData = np.asarray(np.float32(b)/255.0)
                        
                        seg_n = "seg"
#                        領域分割画像の読み込み
                        if os.path.isfile("./data/"+seg_n+"_hall_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img1 = np.array(Image.open("./data/"+seg_n+"_hall_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            a1 = a1 + 1
                        else :
                            seg_img1 = np.array(np.full((0,0), 255, dtype=np.uint8))
            
                        if os.path.isfile("./data/"+seg_n+"_shadow_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img2 = np.array(Image.open("./data/"+seg_n+"_shadow_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            b1 = b1 + 1
                        else :
                            seg_img2 = np.array(np.full((0,0), 255, dtype=np.uint8))
                            
                        if os.path.isfile("./data/"+seg_n+"_hyouzi_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img3 = np.array(Image.open("./data/"+seg_n+"_hyouzi_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            c1 = c1 + 1
                        else :
                            seg_img3 = np.array(np.full((0,0), 255, dtype=np.uint8))
#          
                        seg1  = np.asarray(np.float32(seg_img1)/255.0)
                        seg2  = np.asarray(np.float32(seg_img2)/255.0)
                        seg3  = np.asarray(np.float32(seg_img3)/255.0)
                        
                        imgData = np.asarray([bImgData, gImgData, rImgData,seg1,seg2,seg3])
                        x = imgData



                    if data_channels == 1:
                        grayImgData = np.asarray(np.float32(PIL_data)/255.0)
                        x = grayImgData[None,...]
                    
                    
                    if data_channels == 13: #領域分割結果を合わせて入力

                        grayImgData = np.asarray(np.float32(PIL_data)/255.0)
                        
#                        領域分割画像の読み込み
                        seg_n = "2.5m"
                        if os.path.isfile("./data/"+seg_n+"_hall_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img1 = np.array(Image.open("./data/"+seg_n+"_hall_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                        else :
                            seg_img1 = np.array(np.full((50,50), 0, dtype=np.uint8))
            
                        if os.path.isfile("./data/"+seg_n+"_shadow_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img2 = np.array(Image.open("./data/"+seg_n+"_shadow_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            
                        else :
                            seg_img2 = np.array(np.full((50,50), 0, dtype=np.uint8))
                            
                        if os.path.isfile("./data/"+seg_n+"_hyouzi_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img3 = np.array(Image.open("./data/"+seg_n+"_hyouzi_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                          
                        else :
                            seg_img3 = np.array(np.full((50,50), 0, dtype=np.uint8))
#          
                        seg1  = np.asarray(np.float32(seg_img1)/255.0)
                        seg2  = np.asarray(np.float32(seg_img2)/255.0)
                        seg3  = np.asarray(np.float32(seg_img3)/255.0)
                        
                        
                        imgData = np.asarray([grayImgData,seg1,seg2,seg3])
                        x = imgData

                    
                    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
#                       １か０で出力
                        y = model.predictor(x[None,...]).data.argmax(axis=1)[0]
#                       出力を確率で出す
                        yy = model.predictor(x[None,...])
                        rate = F.softmax(yy.data)[0][1]

                    if y == 1:
#                        入力画像のひび割れ箇所を赤くする
#                        マスクでひび割れ箇所を白くする
                        for y  in range(height_start,height_end):
                            for x  in range(width_start,width_end):
                               mask[y][x] = 255
                               dst_img[y][x][2] = 255 
                                  
#           F値の計算（出力画像，ファイル名，正解画像のフォルダ）       
#           戻り値今ふーじょんマトリックスの値                    
            a,b,c,d = F1_measure(f1_img,mask,file_name,"./data/2.5m_gt_gray_own3/")
            TP = TP + a
            FP = FP + b
            FN = FN + c
            TN = TN + d

        

#    画像データ全体のスコアを計算        
    Precision = (TP)/(TP+FP+0.001)
    Recall = (TP)/(TP+FN+0.001)
    F1 = 2*Recall*Precision/(Recall+Precision)
    Specificity = (TN)/(TN+FP+0.001)      
#   表示
    print ("Precision={:.4}".format(Precision))
    print ("Recall={:.4}".format(Recall))
    print ("Specificity={:.4}".format(Specificity)) 
    print ("F1={:.4}\n\n".format(F1))
#   ファイル書き込み
    f = open("./F1/F1.txt",'w')    
    f.write("Precision={:.4}".format(Precision)+'\n')
    f.write("Recall={:.4}".format(Recall)+"\n")
    f.write("F1={:.4}".format(F1)+'\n')
    f.write("Specificity={:.4}".format(Specificity)+'\n')
    f.close() # ファイルを閉じる
    

   
    return 0    

if __name__ == '__main__':

        predict(1,"gen_seg_median41/")

