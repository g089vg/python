# -*- coding: utf-8 -*-
"""
Created on Tue May 22 13:44:24 2018

@author: g089v
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:33:42 2018

@author: g089v
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:38:23 2017

@author: g089v
"""
import numpy as np
import cv2
import os

 
min_table = 50
max_table = 205
diff_table = max_table - min_table

LUT_HC = np.arange(256, dtype = 'uint8' )
# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255 
 
name = 10
name = str(name)
if __name__ == "__main__":
    data_dir_path = u"./holdout/"+name +"/crack=1_rgb"
    file_list = os.listdir(r"./holdout/"+name +"/crack=1_rgb/")
            


    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            abs_name = data_dir_path + '/' + file_name
#            print(abs_name)
            img = cv2.imread(abs_name)
            dst_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            height,width,channel = img.shape


            for y in range(height):
                for x in range(width):
                    dst_gray[y,x] = max(img[y,x,0],img[y,x,1],img[y,x,2])
                    
            cv2.imwrite("./holdout/"+name +"/crack=1_gray/"+file_name, dst_gray)        
#            cv2.imshow('window', dst_gray)
#            cv2.waitKey(0)

            
#            high_cont_img = cv2.LUT(dst_gray, LUT_HC)   
#            median = cv2.medianBlur(high_cont_img, 5)
            nlm = cv2.fastNlMeansDenoising(dst_gray, None,21,10)            

            imax = dst_gray.max()
              
            dst = dst_gray
#            
            for y in range(height):
                for x in range(width):
                    
                    
#                    dst[y,x] = float(dst_gray[y,x])-float(nlm[y,x])+128
    
                    dst[y,x] = (dst_gray[y,x]*128)/(nlm[y,x])
                    if dst[y,x]<0 :dst[y,x] = 0
                    if dst[y,x]>255 :dst[y,x] = 255
#            nlm = (nlm - np.mean(nlm))/np.std(nlm)*16+64
#            dst = cv2.bitwise_not(dst)
            cv2.imwrite("./holdout/"+name +"/crack=1_nlm/"+file_name, dst)


data_dir_path = u"./holdout/"+name +"/crack=0_rgb"
file_list = os.listdir(r"./holdout/"+name +"/crack=0_rgb/")
            


for file_name in file_list:
    root, ext = os.path.splitext(file_name)
    if ext == u'.bmp':
        abs_name = data_dir_path + '/' + file_name
#        print(abs_name)
        img = cv2.imread(abs_name)
        dst_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        height,width,channel = img.shape


        for y in range(height):
            for x in range(width):
                dst_gray[y,x] = max(img[y,x,0],img[y,x,1],img[y,x,2])
                
        cv2.imwrite("./holdout/"+name +"/crack=0_gray/"+file_name, dst_gray)        
#            cv2.imshow('window', dst_gray)
#            cv2.waitKey(0)

        
#            high_cont_img = cv2.LUT(dst_gray, LUT_HC)   
#            median = cv2.medianBlur(high_cont_img, 5)
        nlm = cv2.fastNlMeansDenoising(dst_gray, None,21,10)            

        imax = dst_gray.max()
          
        dst = dst_gray
#            
        for y in range(height):
            for x in range(width):
                
                
#                    dst[y,x] = float(dst_gray[y,x])-float(nlm[y,x])+128

                dst[y,x] = (dst_gray[y,x]*128)/(nlm[y,x])
                if dst[y,x]<0 :dst[y,x] = 0
                if dst[y,x]>255 :dst[y,x] = 255
#            nlm = (nlm - np.mean(nlm))/np.std(nlm)*16+64
#            dst = cv2.bitwise_not(dst)
        cv2.imwrite("./holdout/"+name +"/crack=0_nlm/"+file_name, dst)





