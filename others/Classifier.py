# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 10:02:15 2018

@author: g089v
"""
import numpy as np
import cv2
import os


def Delete_File (top = '削除したいディレクトリ'):     
  for root, dirs, files in os.walk(top, topdown=False):
      for name in files:
          os.remove(os.path.join(root, name))
      for name in dirs:
          os.rmdir(os.path.join(root, name))



if __name__ == '__main__':
    data_channels = 31
    filter_num = 11
    filter_str = str(filter_num) 
    crack0 = 0
    crack1 = 0
            


    data_dir_rgb = u"./data/half_data/"
    data_dir_gray = u"./data/gray_data/"
    data_dir_m11 = u"./data/median11/"
    data_dir_m21 = u"./data/median21/"
    data_dir_m31 = u"./data/median31/"    
    data_dir_m41 = u"./data/median41/"    
    data_dir_m51 = u"./data/median51/"        
    data_dir_gt = u"./data/t_gt_gray_own/"        
    
    file_list = os.listdir(r'./data/half_data/')
            
    nnum = 0
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            nnum = nnum + 1
            print(file_name)

            file_name = file_name[:-4]


            rgb_img = cv2.imread(data_dir_rgb+file_name+".bmp")
            gray_img = cv2.imread(data_dir_gray+file_name+".bmp")
            median11_img = cv2.imread(data_dir_m11+file_name+".bmp")
            median21_img = cv2.imread(data_dir_m21+file_name+".bmp")
            median31_img = cv2.imread(data_dir_m31+file_name+".bmp")
            median41_img = cv2.imread(data_dir_m41+file_name+".bmp")
            median51_img = cv2.imread(data_dir_m51+file_name+".bmp")
            gt_img = cv2.imread(data_dir_gt+file_name+".bmp",0)
            

            height, width,channels = rgb_img.shape   

#            cv2.imshow("a",mask)
#            cv2.waitKey(0)            
            height_split = 5
            width_split = 7
            new_img_height = int(height / height_split)
            new_img_width = int(width / width_split)
            
            Delete_File("./data/Ground_Truth/"+file_name+"/1/rgb")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/gray/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/median11/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/median21/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/median31/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/median41/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/median51/")           

            Delete_File("./data/Ground_Truth/"+file_name+"/0/rgb/")
            Delete_File("./data/Ground_Truth/"+file_name+"/0/gray/")
            Delete_File("./data/Ground_Truth/"+file_name+"/0/median11/")
            Delete_File("./data/Ground_Truth/"+file_name+"/0/median21/")
            Delete_File("./data/Ground_Truth/"+file_name+"/0/median31/")
            Delete_File("./data/Ground_Truth/"+file_name+"/0/median41/")
            Delete_File("./data/Ground_Truth/"+file_name+"/0/median51/") 
            
            
            
            
            
            num  = 0
            for h in range(height_split):
                height_start = h * new_img_height
                height_end = height_start + new_img_height
        
                for w in range(width_split):
                    
                    width_start = w * new_img_width
                    width_end = width_start + new_img_width
                    num = num +1
                    name = file_name + "_" + str(num) + ".bmp"
#                    print(name)
                    clp_rgb = rgb_img[height_start:height_end, width_start:width_end]
                    clp_gray = gray_img[height_start:height_end, width_start:width_end]
                    clp_m11 = median11_img[height_start:height_end, width_start:width_end]
                    clp_m21 = median21_img[height_start:height_end, width_start:width_end]
                    clp_m31 = median31_img[height_start:height_end, width_start:width_end]
                    clp_m41 = median41_img[height_start:height_end, width_start:width_end]
                    clp_m51 = median51_img[height_start:height_end, width_start:width_end]
                    clp_gt = gt_img[height_start:height_end, width_start:width_end]                    
                    crack = 0
                    
                    for yy in range(new_img_height):
                        for xx in range(new_img_width):
                         if(clp_gt[yy][xx] != 0):crack = crack + 1  
                    if(crack==2500):
                        crack1 = crack1 + 1 
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/1/rgb/"+name, clp_rgb)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/1/gray/"+name, clp_gray)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/1/median11/"+name, clp_m11)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/1/median21/"+name, clp_m21)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/1/median31/"+name, clp_m31)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/1/median41/"+name, clp_m41)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/1/median51/"+name, clp_m51)
                    else : 
                        crack0 = crack0 + 1 
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/0/rgb/"+name, clp_rgb)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/0/gray/"+name, clp_gray)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/0/median11/"+name, clp_m11)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/0/median21/"+name, clp_m21)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/0/median31/"+name, clp_m31)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/0/median41/"+name, clp_m41)
                        cv2.imwrite("./data/Ground_Truth/"+file_name+"/0/median51/"+name, clp_m51)                    
    print(crack0)
    print(crack1)
#2871
#629
                    
