# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:21:06 2018

@author: g089v
"""
import cv2
import os
import pickle
import numpy as np
crack_num = []


def F1_measure(src_image,dst_image,file_name,gt_folda = './data/t_gt_gray_own/',out_name = "./F1/"):
#    正解画像（ひび割れ発生箇所のマスク画像）
    gt_img = cv2.imread(gt_folda+file_name+'.bmp',0)
#   正検出，過検出，未検出出力用
    f1_img = src_image

    height, width = gt_img.shape[0],gt_img.shape[1]
    

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    
    
    height_split = 5
    width_split = 7
    new_img_height = int(height / height_split)
    new_img_width = int(width / width_split)
   

    for h in range(height_split):
        height_start = h * new_img_height
        height_end = height_start + new_img_height

        for w in range(width_split):
            
            width_start = w * new_img_width
            width_end = width_start + new_img_width
    
#            分割した画像ごとに正解を判断(255がひび割れ)
            
            for y  in range(height_start,height_end):
                for x  in range(width_start,width_end):    

                    
                    if gt_img[y][x] == 255 and dst_image[y][x] == 255:
                            cv2.rectangle(f1_img,(width_start+1,height_start+1),(width_end-1,height_end-1),(0,255,0))
                            TP = TP + 1
                    if gt_img[y][x] == 0 and dst_image[y][x] == 0:
                            TN = TN + 1
                        
                    if gt_img[y][x] == 255 and dst_image[y][x] == 0:
                            cv2.rectangle(f1_img,(width_start+1,height_start+1),(width_end-1,height_end-1),(255,0,0))
                            FN = FN + 1
                             
                    if gt_img[y][x] == 0 and dst_image[y][x] == 255:
                           cv2.rectangle(f1_img,(width_start+1,height_start+1),(width_end-1,height_end-1),(0,0,255))
                           FP = FP + 1

                

    Precision = (TP+0.001)/(TP+FP+0.001)
    Recall = (TP+0.001)/(TP+FN+0.001)
    Specificity = (TN+0.001)/(TN+FP+0.001)
    F1 = 2*Recall*Precision/(Recall+Precision)
      
    print ("Precision={:.4}です".format(Precision))
    print ("Recall={:.4}です".format(Recall))
    print ("Specificity={:.4}です".format(Specificity))
    print ("F1={:.4}です\n\n".format(F1))
               
    cv2.imwrite(out_name+file_name+'.bmp', f1_img)    
#    print(out_name+file_name+'.bmp')                
    return TP,FP,FN,TN
