
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 08:42:50 2018
@author: g089v
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from random import random as rand
from PIL import Image
from Zhang_Suen import Zhang_Suen_thinning
from Zhang_Suen import Draw_Line
#import Zhang_Suen
def Delete_File (top = '削除したいディレクトリ'):     
  for root, dirs, files in os.walk(top, topdown=False):
      for name in files:
          os.remove(os.path.join(root, name))
      for name in dirs:
          os.rmdir(os.path.join(root, name))
          

def transform(img,center):
#画像サイズの取得
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    
    #水滴を落としたあとの画像として、元画像のコピーを作成。後処理で
    img2 = img.copy()
    
    #水滴の中心と半径の指定

    r = random.randint(20, 40)
    
    #ピクセルの座標を変換
    for x in range(width):
        for y in range(height):
            #dはこれから処理を行うピクセルの、水滴の中心からの距離
            d = np.linalg.norm(center - np.array((y,x)))
    
            #dが水滴の半径より小さければ座標を変換する処理をする
            if d < r:
                #vectorは変換ベクトル。説明はコード外で。
                vector = (d / r)**1.4 * (np.array((y,x)) - center)
    
                #変換後の座標を整数に変換
                p = (center + vector).astype(np.int32)
    
                #色のデータの置き換え            
                img2[y,x]=img[p[0],p[1]]
    return(img2)

def Label(img):
    img1=np.asarray(img)
#    print(img.type)
    # ラベリング処理
    mono_src = cv2.threshold(img1, 200, 255, cv2.THRESH_BINARY_INV)[1]
    
    
#    im_list = np.asarray(mono_src)
#    #貼り付け
#    plt.imshow(im_list)
#    #表示
#    plt.show()   
    
    ret, markers = cv2.connectedComponents(mono_src)
    label = cv2.connectedComponentsWithStats(mono_src)  
    
    
    # ラベリング結果書き出し準備
    height, width = mono_src.shape[:2]
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    s_max = 0

    for i in range(n): 
#        print(data[i][4])

        if s_max < data[i][4]  :
            s_max = data[i][4]

    return(s_max)


if __name__ == '__main__':
    # 3.3.0
    num = 0
    h = 50
    w = 50
    color = 10
    thin = 1
    size = (h,w)
    
    create_image = 6 
    img_line = []
    img_ = np.full(size, 255, dtype=np.uint8)
    img_line = Draw_Line(img_,0,1)
    
    cv2.imwrite("./data/generate_image2/a.bmp", img_line[1])
#    img.append([img_])

#    Delete_File("./data/generate_image1/")
#    Delete_File("./data/generate_image2/")

    file_list = os.listdir(r"./data/NO_CRACK_DATA/")
    mode = "rgb"
    channels = 0
    number = 0.0
    
    
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':

            
            inp = []
            for num in range (create_image):
                inp.append(cv2.imread("./data/NO_CRACK_DATA/"+file_name))


            ave1 = 0.0
            ave2 = 0.0
            ave3 = 0.0
            for y in range (h) :
                for x in range( w) :
                    ave1  = inp[0][y][x][0] + ave1
                    ave2  = inp[0][y][x][1] + ave2
                    ave3  = inp[0][y][x][2] + ave3
            ave1  = ave1 / float(h*w)
            ave2  = ave2 / float(h*w)
            ave3  = ave3 / float(h*w)   
            ave = (ave1+ave2+ave3)/3.0
            print("average={:.4}\n\n".format(ave))
            file_name = file_name[:-4]
            

            trm = []
            blur = []
            center_ = 0
            for num in range (create_image):
                T = 0
                while T == 0:
                    center_ = (random.randint(0, 49),random.randint(0, 49))
#                    print(center_) 

                    if img_line[num][center_[0]][center_[1]] == 255: 
                        T = 1

                trm.append(transform(img_line[num],center_))
                blur.append(cv2.threshold(trm[num], 150, 255, cv2.THRESH_BINARY)[1])
            
            cv2.imwrite("./data/generate_image2/b.bmp", blur[1])

        
            length = random.randint(1, 10)
            print(length)
            epsilon = 0.5
            if epsilon > rand():   
                for num in range (create_image):
                    
                    for yy in range(h):
                        for xx in range(w):
                            if(xx<length or xx > w-length or yy < length or yy > h-length):
                                blur[num][yy][xx] = 255
                            else:
                                blur[num][yy][xx] = blur[num][yy][xx]

                
#            ラベリングで面積を取得  
            s = []
            out = []
            for num in range (create_image):       
                s.append(Label(blur[num]))   
                epsilon = 0.7
                if epsilon > rand():                  
                    out.append(blur[num])
                else:
                    out.append(Zhang_Suen_thinning(blur[num]))


            
            im_list = np.asarray(out[0])
            #貼り付け
            plt.imshow(im_list)
            #表示
            plt.show()
            print("S="+str(s[0]))
            rate = rand()
            epsilon = 0.7
            
            color1 = ave -80
            if(color1 < 0):color1 =10
            if(ave < 20):color1 =50  
            
            color2 = ave + 150
            if(color2 > 255):color2 =255
            if(ave > 140):color2 =10 
            
            
            for num in range (create_image):   
                
                for y in range(h):
                    for x in range(w):
                        
                        if(out[num][y][x] == 0):
    
                            if epsilon > rate:   
                                inp[num][y][x] = color1

                            else: 
                                inp[num][y][x] = color2
            cv2.imwrite("./data/generate_image2/d.bmp", inp[1])

            for num in range (create_image):     
                      
                inp[num] = cv2.GaussianBlur(inp[num],(5,5),0)             
                if s[num] >= 10 and s[num]<=400:cv2.imwrite("./data/generate_image1/"+file_name+"_"+str(num)+".bmp", inp[num])
            cv2.imwrite("./data/generate_image2/e.bmp", inp[1])

            number = number +1
