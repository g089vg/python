# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:21:06 2018

@author: g089v
"""
import cv2

def F1_measure(src_image,dst_image,file_name,gt_folda = './gt_image/'):
    gt_img = cv2.imread(gt_folda+file_name+'.bmp')
#    print(gt_folda+file_name+'.bmp')
    f1_img = src_image

    height, width,channels = gt_img.shape
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    
    for y  in range(height-1):
        for x  in range(width-1):
            if dst_image[y][x][0]== gt_img[y][x][0] and dst_image[y][x][1]== gt_img[y][x][1] and dst_image[y][x][2]==gt_img[y][x][2]:
                if gt_img[y][x][2] == 255 and gt_img[y][x][0] < 200 and gt_img[y][x][1] < 200:
                    f1_img[y][x][1] = 128
                    TP = TP + 1
                else:
                    TN = TN + 1
            else :
                 if gt_img[y][x][2] == 255 and gt_img[y][x][0] < 200 and gt_img[y][x][1] < 200:
                     f1_img[y][x][0] = 255
                     FN = FN + 1
                     
                 if dst_image[y][x][2] == 255 and gt_img[y][x][0] < 200 and gt_img[y][x][1] < 200:
                     f1_img[y][x][2] = 255     
                     FP = FP + 1
                     
    Precision = TP/(TP+FP+0.1)
    Recall = TP/(TP+FN+0.1)
    F1 = 2*Recall*Precision/(Recall+Precision+0.1)
      
    print ("Precision={:.4}です".format(Precision))
    print ("Recall={:.4}です".format(Recall))
    print ("F1={:.4}です\n\n".format(F1))
#    cv2.imshow('a',f1_img)
#    cv2.waitKey(0)                
    cv2.imwrite('./F1/'+file_name+'.bmp', f1_img)                    
    return TP,FP,FN,TN