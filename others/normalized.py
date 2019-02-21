# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:38:23 2017

@author: g089v
"""
import numpy as np
import cv2
import os



def rgb_to_gray(src):
     # チャンネル分解
     r, g, b = src[:,:,0], src[:,:,1], src[:,:,2]
     # R, G, Bの値からGrayの値に変換
     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
     
     return gray   
 
if __name__ == "__main__":
    data_dir_path = u"./2.5m/"
    file_list = os.listdir(r'./2.5m/')
            


    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.jpg':
            abs_name = data_dir_path + '/' + file_name


            img = cv2.imread(abs_name,cv2.IMREAD_UNCHANGED)
            gray = rgb_to_gray(img)
            print(abs_name)
            height, width = img.shape[:2]
            channels = 1
            ave = np.average(gray)
            print(ave)
            img = cv2.GaussianBlur(img, ksize=(3,3), sigmaX=1.3)
            bi3 = (img - np.mean(img))/np.std(img)*16+20
 
            cv2.imwrite("./normalized/"+file_name, bi3)
#            if ave > 110 :
#                img = (img - np.mean(img))/np.std(img)*16+160
#                cv2.imwrite("./normalized/"+file_name, img)
#            else :
#                cv2.imwrite("./normalized/"+file_name, img)
