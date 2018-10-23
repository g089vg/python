# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 09:32:50 2018

@author: g089v
"""
import cv2
import numpy as np
# Zhang-Suenのアルゴリズムを用いて2値化画像を細線化します
def Draw_Line(img,color,thin):
    
    img_ = []
    for num in range(100):    
        img_.append(img.copy())

    
    img_out = []    
    
    img_out.append(cv2.line(img_[0], (0, 0), (50, 50), color, thickness=thin, lineType=cv2.LINE_AA))#斜め右   
    
    img_out.append(cv2.line(img_[2], (0, 50), (50, 0), color, thickness=thin, lineType=cv2.LINE_AA))#斜め左
    
    img_out.append(cv2.line(img_[3], (25, 0), (25, 50), color, thickness=thin, lineType=cv2.LINE_AA))#縦
    
    img_out.append(cv2.line(img_[4], (0, 25), (50, 25), color, thickness=thin, lineType=cv2.LINE_AA) )#横  
    

    a = (cv2.line(img_[5], (10, 0), (10, 50), color, thickness=thin, lineType=cv2.LINE_AA))#縦2
    img_out.append(cv2.line(a, (35, 0), (35, 50), color, thickness=thin, lineType=cv2.LINE_AA))#縦2


    b = (cv2.line(img_[6], (15, 25), (15, 25), color, thickness=thin, lineType=cv2.LINE_AA) )#横2
    img_out.append(cv2.line(b, (35, 25), (35, 25), color, thickness=thin, lineType=cv2.LINE_AA) )#横 2
    
#    c = (cv2.line(img_[5], (10, 0), (10, 50), color, thickness=thin, lineType=cv2.LINE_AA))#縦3
#    c = (cv2.line(c, (20, 0), (20, 50), color, thickness=thin, lineType=cv2.LINE_AA))#縦3    
#    img_out.append(cv2.line(c, (35, 0), (35, 50), color, thickness=thin, lineType=cv2.LINE_AA))#縦3
#    
#    d = (cv2.line(img_[6], (15, 25), (15, 25), color, thickness=thin, lineType=cv2.LINE_AA) )#横2
#    d = cv2.line(d, (25, 25), (25, 25), color, thickness=thin, lineType=cv2.LINE_AA) #横 2    
#    img_out.append(cv2.line(d, (35, 25), (35, 25), color, thickness=thin, lineType=cv2.LINE_AA) )#横 2

    
    return img_out  


def Zhang_Suen_thinning(binary_image):
    
    
    
    
    
    
    
    
  
    ret,th2 = cv2.threshold(binary_image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)    
    th2 = padding_zeros(th2)
    image_thinned = black_one(th2)    
    

    # 初期化します。この値は次のwhile文の中で除かれます。
    changing_1 = changing_2 = [1]
    while changing_1 or changing_2:
        # ステップ1
        changing_1 = []
        rows, columns = image_thinned.shape
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and # 条件2
                    count_transition(neighbour_points) == 1 and # 条件3
                    p2 * p4 * p6 == 0 and # 条件4
                    p4 * p6 * p8 == 0): # 条件5
                    changing_1.append((x,y))
        for x, y in changing_1:
            image_thinned[x][y] = 0
        # ステップ2
        changing_2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns -1):
                p2, p3, p4, p5, p6, p7, p8, p9 = neighbour_points = neighbours(x, y, image_thinned)
                if (image_thinned[x][y] == 1 and
                    2 <= sum(neighbour_points) <= 6 and # 条件2
                    count_transition(neighbour_points) == 1 and # 条件3
                    p2 * p4 * p8 == 0 and # 条件4
                    p2 * p6 * p8 == 0): # 条件5
                    changing_2.append((x,y))
        for x, y in changing_2:
            image_thinned[x][y] = 0        
    image_thinned = inv_black_one(unpadding(image_thinned))
    OpenCV_data=np.asarray(image_thinned)
    return OpenCV_data

# 2値画像の黒を1、白を0とするように変換するメソッドです
def black_one(binary):
    bool_image = binary.astype(bool)
    inv_bool_image = ~bool_image
    return inv_bool_image.astype(int)

# 画像の外周を0で埋めるメソッドです
def padding_zeros(image):
    import numpy as np
    m,n = np.shape(image)
    padded_image = np.zeros((m+2,n+2))
    padded_image[1:-1,1:-1] = image
    return padded_image

# 外周1行1列を除くメソッドです。
def unpadding(image):
    return image[1:-1, 1:-1]

# 指定されたピクセルの周囲のピクセルを取得するメソッドです
def neighbours(x, y, image):
    return [image[x-1][y], image[x-1][y+1], image[x][y+1], image[x+1][y+1], # 2, 3, 4, 5
             image[x+1][y], image[x+1][y-1], image[x][y-1], image[x-1][y-1]] # 6, 7, 8, 9

# 0→1の変化の回数を数えるメソッドです
def count_transition(neighbours):
    neighbours += neighbours[:1]
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(neighbours, neighbours[1:]) )

# 黒を1、白を0とする画像を、2値画像に戻すメソッドです
def inv_black_one(inv_bool_image):
    bool_image = ~inv_bool_image.astype(bool)
    return bool_image.astype(int) * 255