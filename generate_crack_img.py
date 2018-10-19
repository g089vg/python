# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 08:42:50 2018

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:03:27 2018

@author: mitani
"""

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



def transform(img,center):
#画像サイズの取得
    height = np.shape(img)[0]
    width = np.shape(img)[1]
    
    #水滴を落としたあとの画像として、元画像のコピーを作成。後処理で
    img2 = img.copy()
    
    #水滴の中心と半径の指定

    r = 40
    
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
    

    # ラベリング処理
    mono_src = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)[1]
    
    
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
    thin = 2
    size = (h,w)
    img1 = np.full(size, 255, dtype=np.uint8)
    img2 = np.full(size, 255, dtype=np.uint8)
    img3 = np.full(size, 255, dtype=np.uint8)
    img4 = np.full(size, 255, dtype=np.uint8)   
    img1 = cv2.line(img1, (0, 0), (50, 50), color, thickness=thin, lineType=cv2.LINE_AA)
    img2 = cv2.line(img2, (0, 50), (50, 0), color, thickness=thin, lineType=cv2.LINE_AA)
    img3 = cv2.line(img3, (25, 0), (25, 50), color, thickness=thin, lineType=cv2.LINE_AA)
    img4 = cv2.line(img4, (0, 25), (50, 25), color, thickness=thin, lineType=cv2.LINE_AA)


    file_list = os.listdir(r"./data/NO_CRACK_DATA/")
    mode = "rgb"
    channels = 0
    num = 0.0
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            rate = float(num/2871.0)
            print("load_rate={:.4}%\n\n".format(rate*100))
            inp1 = cv2.imread("./data/NO_CRACK_DATA/"+file_name)
            inp2 = cv2.imread("./data/NO_CRACK_DATA/"+file_name)
            inp3 = cv2.imread("./data/NO_CRACK_DATA/"+file_name)
            inp4 = cv2.imread("./data/NO_CRACK_DATA/"+file_name)

            ave1 = 0.0
            ave2 = 0.0
            ave3 = 0.0
            for y in range (h) :
                for x in range( w) :
                    ave1  = inp1[y][x][0] + ave1
                    ave2  = inp1[y][x][1] + ave2
                    ave3  = inp1[y][x][2] + ave3
            ave1  = ave1 / float(h*w)
            ave2  = ave2 / float(h*w)
            ave3  = ave3 / float(h*w)   
            ave = (ave1+ave2+ave3)/3.0
            print(ave)
            file_name = file_name[:-4]
            
            T = 0
            while T == 0:
                center1 = random.randint(0, 49),random.randint(0, 49)
                if img1[center1] == 255: 
                    T = 1
            T = 0
            while T == 0:
                center2 = random.randint(0, 49),random.randint(0, 49)
                if img2[center2] == 255: 
                    T = 1
            T = 0
            while T == 0:
                center3 = random.randint(0, 49),random.randint(0, 49)
                if img3[center3] == 255: 
                    T = 1
            T = 0
            while T == 0:
                center4 = random.randint(0, 49),random.randint(0, 49)
                if img4[center4] == 255: 
                    T = 1
            blur1 = transform(img1,center1)
            blur2 = transform(img2,center2) 
            blur3 = transform(img3,center3)
            blur4 = transform(img4,center4)            
            
#            blur1 = cv2.GaussianBlur(transform(img1,center1),(7,7),0) 
#            blur2 = cv2.GaussianBlur(transform(img2,center2),(7,7),0) 
#            blur3 = cv2.GaussianBlur(transform(img3,center3),(7,7),0) 
#            blur4 = cv2.GaussianBlur(transform(img4,center4),(7,7),0) 

            s1 = Label(blur1) 
            s2 = Label(blur2) 
            s3 = Label(blur3) 
            s4 = Label(blur4) 

            blur1 = cv2.threshold(blur1, 100, 255, cv2.THRESH_BINARY)[1]
            blur2 = cv2.threshold(blur2, 100, 255, cv2.THRESH_BINARY)[1]
            blur3 = cv2.threshold(blur3, 100, 255, cv2.THRESH_BINARY)[1]
            blur4 = cv2.threshold(blur4, 100, 255, cv2.THRESH_BINARY)[1]
            im_list = np.asarray(blur1)
            #貼り付け
            plt.imshow(im_list)
            #表示
            plt.show()


#            print(s1,s2,s3,s4)
            
            out1 = np.full(size, 255, dtype=np.uint8)
            out2 = np.full(size, 255, dtype=np.uint8)
            out3 = np.full(size, 255, dtype=np.uint8)
            out4 = np.full(size, 255, dtype=np.uint8)
        
#            length = random.randint(5, 15)
#            epsilon = 0.5
#            if epsilon > rand():   
#                for yy in range(length,h-length):
#                    for xx in range(length,w-length):
#                        out1[yy][xx] = blur1[yy][xx]
#                        out2[yy][xx] = blur2[yy][xx]
#                        out3[yy][xx] = blur3[yy][xx]
#                        out4[yy][xx] = blur4[yy][xx]
#            else:
#                out1 = blur1
#                out2 = blur2
#                out3 = blur3
#                out4 = blur4
            out1 = blur1
            out2 = blur2
            out3 = blur3
            out4 = blur4           
            color = ave -30
            if(color < 0):color =10             
            for y in range(h):
                for x in range(w):
                    
                    if(out1[y][x] == 0):
                        inp1[y][x][0] = color
                        inp1[y][x][1] = color
                        inp1[y][x][2] = color
                    if(out2[y][x] == 0):                        
                        inp2[y][x][0] = color
                        inp2[y][x][1] = color
                        inp2[y][x][2] = color
                    if(out3[y][x] == 0):                        
                        inp3[y][x][0] = color
                        inp3[y][x][1] = color
                        inp3[y][x][2] = color
                    if(out4[y][x] == 0):                        
                        inp4[y][x][0] = color
                        inp4[y][x][1] = color
                        inp4[y][x][2] = color
            inp1 = cv2.GaussianBlur(inp1,(3,3),0) 
            inp2 = cv2.GaussianBlur(inp2,(3,3),0) 
            inp3 = cv2.GaussianBlur(inp3,(3,3),0) 
            inp4 = cv2.GaussianBlur(inp4,(3,3),0) 
            
            if s1 >= 100 and s1<=400:cv2.imwrite("./data/generate_image/"+file_name+"_1.bmp", inp1)
            if s2 >= 100 and s2<=400:cv2.imwrite("./data/generate_image/"+file_name+"_2.bmp", inp2)
            if s3 >= 100 and s3<=400:cv2.imwrite("./data/generate_image/"+file_name+"_3.bmp", inp3)
            if s4 >= 100 and s4<=400:cv2.imwrite("./data/generate_image/"+file_name+"_4.bmp", inp4)
            num = num +1
