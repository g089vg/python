# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:16:11 2018

@author: g089v
"""

import numpy as np
import cv2
import os

  
 
if __name__ == "__main__":
    data_dir_path1 = u"./half_data/"
#    data_dir_path2 = u"./gt_half"
    file_list = os.listdir(r'./half_data/')
            

    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            
            
            abs_name = data_dir_path1 + '/' + file_name
#            true_name = data_dir_path2 + '/' + file_name

            file_name = file_name[:-4]
            img = cv2.imread(abs_name,0)
#            true_img = cv2.imread(true_name,1)

            height, width = img.shape
        
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
                    name0 = "./batch_image/"+file_name + "_" + str(num) + ".bmp"
#                    name1 = "./crack_1/"+file_name + "_" + str(num) + ".bmp"
#                    print(name)
                    clp1 = img[height_start:height_end, width_start:width_end]
#                    clp2 = true_img[height_start:height_end, width_start:width_end]
                    crack = 0
                    cv2.imwrite(name0, clp1)
#                    cv2.imshow("0",clp1)
#                    cv2.imshow("1",clp2)
#                    cv2.waitKey(0)
#                    for y in range(new_img_height):
#                        for x in range(new_img_width):
#                            if (clp2[y,x,0] == 255):
#                                crack = crack + 1
                                
#                    img_resize = cv2.resize(clp2,(clp2.shape[1]*10,clp2.shape[0]*10),interpolation=cv2.INTER_LINEAR)
#                    cv2.imshow("image",img_resize)
#
#                    while (True):
#                        if cv2.waitKey(1) & 0xFF == ord("0"):
#                            cv2.imwrite(name0, clp1)
#                            break
#                        
#                        if cv2.waitKey(1) & 0xFF == ord("1"):
#                            cv2.imwrite(name1, clp1)
#                            break
#                    cv2.destroyAllWindows()
#                    print(crack)
#                    if crack >= 50 : cv2.imwrite(name1, clp1)
