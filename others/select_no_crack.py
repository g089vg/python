# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 12:01:20 2018

@author: g089v
"""
import os
import cv2
from random import random as rand
file_list = os.listdir(r"./data/half_data/")
mode = "rgb"
channels = 3

def Delete_File (top = '削除したいディレクトリ'):     
  for root, dirs, files in os.walk(top, topdown=False):
      for name in files:
          os.remove(os.path.join(root, name))
      for name in dirs:
          os.rmdir(os.path.join(root, name))
          
Delete_File("./data/NO_CRACK_DATA/")
for file_name in file_list:
    root, ext = os.path.splitext(file_name)
    if ext == u'.bmp':
        file_name =  file_name[:-4]
        print(file_name)
        

        
        file_list0 = os.listdir(r"./data/Ground_Truth/"+file_name+"/0/"+mode+"/")    
        data_dir_path0 = u"./data/Ground_Truth/"+file_name+"/0/"+mode+"/"
        outputname0 = "./data/Ground_Truth/"+file_name+"/0/inf_"+mode+"/"
        file_list2 = file_list0
        data_dir_path2 = data_dir_path0
        outputname2 = outputname0
        for file_name2 in file_list2:
            root, ext = os.path.splitext(file_name2)
            if ext == u'.bmp':
                
                if os.path.isfile("./data/seg_hyouzi_batch/"+file_name2) == False:
                        
                    print("1----"+data_dir_path2+file_name2)
                    img = cv2.imread(data_dir_path2+file_name2)
                    cv2.imwrite("./data/NO_CRACK_DATA/"+file_name2,img)
