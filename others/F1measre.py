# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:21:06 2018

@author: g089v
"""
import cv2

def F1_measure(src_image,dst_image,file_name,segmentation = 0,gt_folda = './data/t_gt_gray_own/'):
    gt_img = cv2.imread(gt_folda+file_name+'.bmp',0)
    
    mask1 = cv2.imread("./data/Mask_50/"+file_name+'.bmp',0)
#    print(gt_folda+file_name+'.bmp')
    f1_img = src_image
#    cv2.imshow('a',dst_image)
#    cv2.waitKey(0) 
    height, width = gt_img.shape
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    
    for y  in range(height):
        for x  in range(width):
            
            if segmentation == 1:
                if mask1[y][x] == 0:   
                    if gt_img[y][x] == 255 and dst_image[y][x] == 255:
                            f1_img[y][x][1] = 128
                            TP = TP + 1
                    if gt_img[y][x] == 0 and dst_image[y][x] == 0:
                            TN = TN + 1
                            
                    if gt_img[y][x] == 255 and dst_image[y][x] == 0:
                         f1_img[y][x][0] = 255
                         FN = FN + 1
                             
                    if gt_img[y][x] == 0 and dst_image[y][x] == 255:
                     f1_img[y][x][2] = 255     
                     FP = FP + 1
                else :
                     f1_img[y][x][2] = 255
                     f1_img[y][x][1] = 255
            else:
                if gt_img[y][x] == 255 and dst_image[y][x] == 255:
                        f1_img[y][x][1] = 128
                        TP = TP + 1
                if gt_img[y][x] == 0 and dst_image[y][x] == 0:
                        TN = TN + 1
                        
                if gt_img[y][x] == 255 and dst_image[y][x] == 0:
                     f1_img[y][x][0] = 255
                     FN = FN + 1
                         
                if gt_img[y][x] == 0 and dst_image[y][x] == 255:
                 f1_img[y][x][2] = 255     
                 FP = FP + 1

    Precision = (TP+0.001)/(TP+FP+0.001)
    Recall = (TP+0.001)/(TP+FN+0.001)
    Specificity = (TN+0.001)/(TN+FP+0.001)
    F1 = 2*Recall*Precision/(Recall+Precision)
      
    print ("Precision={:.4}です".format(Precision))
    print ("Recall={:.4}です".format(Recall))
    print ("Specificity={:.4}です".format(Specificity))
    print ("F1={:.4}です\n\n".format(F1))
               
    cv2.imwrite('./F1/'+file_name+'.bmp', f1_img)                    
    return TP,FP,FN,TN
