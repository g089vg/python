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



def Delete_File (top = '削除したいディレクトリ'):     
  for root, dirs, files in os.walk(top, topdown=False):
      for name in files:
          os.remove(os.path.join(root, name))
      for name in dirs:
          os.rmdir(os.path.join(root, name))

    
if __name__ == "__main__":
    file_list = os.listdir(r"./data/half_data/")
    crack0  = 0
    crack1  = 0
    
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            file_name =  file_name[:-4]
            print(file_name)
            
            file_list0 = os.listdir("./data/Ground_Truth/"+file_name+"/0/inf_rgb/")
            file_list1 = os.listdir("./data/Ground_Truth/"+file_name+"/1/inf_rgb/")
            data_dir_path0 = u"./data/Ground_Truth/"+file_name+"/0/inf_rgb/"
            data_dir_path1 = u"./data/Ground_Truth/"+file_name+"/1/inf_rgb/"       
      
            outputname0 = "./data/Ground_Truth/"+file_name+"/0/inf_"
            outputname1 = "./data/Ground_Truth/"+file_name+"/1/inf_"
            
            Delete_File("./data/Ground_Truth/"+file_name+"/0/inf_gray/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/inf_median11/")         
            Delete_File("./data/Ground_Truth/"+file_name+"/1/inf_median21/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/inf_median31/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/inf_median41/")
            Delete_File("./data/Ground_Truth/"+file_name+"/1/inf_median51/")
               
            for xx in range(2):
                file_list2 = []
                data_dir_path2 = []
                outputname2 = []
                
                if xx == 0:
    #                print("crack")
                    crack0  = crack0 + 1
                    file_list2 = file_list0
                    data_dir_path2 = data_dir_path0
                    outputname2 = outputname0
                if xx == 1:
    #                print("no crack")
                    crack1  = crack1 + 1
                    file_list2 = file_list1
                    data_dir_path2 = data_dir_path1
                    outputname2 = outputname1
                    
                for file_name2 in file_list2:
                    root, ext = os.path.splitext(file_name2)
                    if ext == u'.bmp':
#                        print(file_name2)

                        img = cv2.imread(data_dir_path2+file_name2)
                        dst_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        dst = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        height,width = 50,50

                        for y in range(height):
                            for x in range(width):
                                dst[y][x] = 255
            
                        for y in range(height):
                            for x in range(width):
                                dst_gray[y,x] = max(img[y,x,0],img[y,x,1],img[y,x,2])
                                
                        cv2.imwrite(outputname2+"gray/"+file_name2, dst_gray)        

                        dst11 = dst_gray.copy()  
                        dst21 = dst_gray.copy() 
                        dst31 = dst_gray.copy() 
                        dst41 = dst_gray.copy() 
                        dst51 = dst_gray.copy() 
                
         
                        median11 = cv2.medianBlur(dst_gray, 11)
                        median21 = cv2.medianBlur(dst_gray, 21)
                        median31 = cv2.medianBlur(dst_gray, 31)
                        median41 = cv2.medianBlur(dst_gray, 41)
                        median51 = cv2.medianBlur(dst_gray, 51)
#                        nlm = cv2.fastNlMeansDenoising(dst_gray, None,filter_num,10)            
#                        cv2.imshow('window', median)
#                        cv2.waitKey(3)    
                        imax = dst_gray.max()
                  
              
                            
                        for y in range(height):
                            for x in range(width):                                
            #                    dst[y,x] = float(dst_gray[y,x])-float(nlm[y,x])+128
                                a11 = float(dst_gray[y,x])/float(median11[y,x]+0.01)*128
                                a21 = float(dst_gray[y,x])/float(median21[y,x]+0.01)*128
                                a31 = float(dst_gray[y,x])/float(median31[y,x]+0.01)*128
                                a41 = float(dst_gray[y,x])/float(median41[y,x]+0.01)*128
                                a51 = float(dst_gray[y,x])/float(median51[y,x]+0.01)*128
        
                                if a11<0.0 or a21<0.0 or a31<0.0 or a41<0.0 or a51<0.0 :
                                    dst11[y,x] = 0
                                    dst21[y,x] = 0
                                    dst31[y,x] = 0
                                    dst41[y,x] = 0
                                    dst51[y,x] = 0
        
                                elif a11>254.0 or a21>254.0 or a31>254.0 or a41>254.0 or a51>254.0:
                                    dst11[y,x] = 255
                                    dst21[y,x] = 255
                                    dst31[y,x] = 255
                                    dst41[y,x] = 255
                                    dst51[y,x] = 255
                                else : 
                                    dst11[y,x] = int(a11)   
                                    dst21[y,x] = int(a21)
                                    dst31[y,x] = int(a31)
                                    dst41[y,x] = int(a41)
                                    dst51[y,x] = int(a51)
                        cv2.imwrite(outputname2+"median11/"+file_name2, dst11)        
                        cv2.imwrite(outputname2+"median21/"+file_name2, dst21)   
                        cv2.imwrite(outputname2+"median31/"+file_name2, dst31)   
                        cv2.imwrite(outputname2+"median41/"+file_name2, dst41)   
                        cv2.imwrite(outputname2+"median51/"+file_name2, dst51)   


    print(crack0)
    print(crack1)

