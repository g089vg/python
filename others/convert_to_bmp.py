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

import cv2
import os

  
 
if __name__ == "__main__":
    data_dir_path = u"./2.5m"
    file_list = os.listdir(r'./2.5m/')
            


    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            img = cv2.imread(abs_name,cv2.IMREAD_UNCHANGED)
            dst = img[250:500,0:350]
#            file_name = file_name[:-10]
            print(file_name)

            file_name = file_name[:-4]
            cv2.imwrite("./2.5m_half/"+file_name+".bmp", dst)
