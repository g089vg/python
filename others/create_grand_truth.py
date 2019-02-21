# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 09:19:17 2018

@author: g089v
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:16:11 2018

@author: g089v
"""

import cv2
import os
import numpy as np
  

class mouseParam:
    def __init__(self, input_img_name):
        #マウス入力用のパラメータ
        self.mouseEvent = {"x":None, "y":None, "event":None, "flags":None}
        #マウス入力の設定
        cv2.setMouseCallback(input_img_name, self.__CallBackFunc, None)
    
    #コールバック関数
    def __CallBackFunc(self, eventType, x, y, flags, userdata):
        
        self.mouseEvent["x"] = x
        self.mouseEvent["y"] = y
        self.mouseEvent["event"] = eventType    
        self.mouseEvent["flags"] = flags    

    #マウス入力用のパラメータを返すための関数
    def getData(self):
        return self.mouseEvent
    
    #マウスイベントを返す関数
    def getEvent(self):
        return self.mouseEvent["event"]                

    #マウスフラグを返す関数
    def getFlags(self):
        return self.mouseEvent["flags"]                

    #xの座標を返す関数
    def getX(self):
        return self.mouseEvent["x"]  

    #yの座標を返す関数
    def getY(self):
        return self.mouseEvent["y"]  

    #xとyの座標を返す関数
    def getPos(self):
        return (self.mouseEvent["x"], self.mouseEvent["y"])
    
def Label():
    num = 0
    height_split = 5
    width_split = 7
    height = 250
    width = 350
    pixel_label = np.zeros((width, height))
    gt_label = np.zeros((width, height))
      
    new_img_height = int(height / height_split)
    new_img_width = int(width / width_split)
    
    for h in range(height_split):
        height_start = h * new_img_height
        height_end = height_start + new_img_height      
        for w in range(width_split):
            width_start = w * new_img_width      
            width_end = width_start + new_img_width
        
            for y in range(height_start,height_end):
                for x in range(width_start,width_end):
                    pixel_label[x][y] = num
            num = num + 1    
    return(pixel_label,gt_label)
    
if __name__ == "__main__":
    data_dir_path1 = u"./data/half_data"
#    data_dir_path2 = u"./gt_half"
    file_list = os.listdir(r'./data/half_data/')
            

        
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            name0 = "./gt_image/"+file_name
            abs_name = data_dir_path1 + '/' + file_name
            img = cv2.imread(abs_name)
            dst = cv2.imread(abs_name)
            print(abs_name)

            height, width,channels = img.shape       

            num  = 0
           
            window_name = "image window"
            cv2.imshow(window_name, dst)
            mouseData = mouseParam(window_name)
            p_label,gt_label = Label()
            while 1:
                cv2.waitKey(100)
                #左クリックがあったら表示
                if mouseData.getEvent() == cv2.EVENT_LBUTTONDOWN:
                    p_x,p_y = mouseData.getPos()
                    label = p_label[p_x][p_y]
                    print(label)
                    for y  in range(height-1):
                        for x  in range(width-1):
                            
                            if p_label[x][y] == label and gt_label[x][y] == 0:
                                gt_label[x][y] = 1
                                
                            elif p_label[x][y] == label and gt_label[x][y] == 1:
                                gt_label[x][y] = 0                                
                            
                    for y  in range(height):
                        for x  in range(width):
                            
                            if gt_label[x][y] == 1:
                                dst[y][x][0] = img[y][x][0]
                                dst[y][x][1] = img[y][x][1]
                                dst[y][x][2] = 255
                            if gt_label[x][y] == 0:
                                dst[y][x][0] = img[y][x][0]
                                dst[y][x][1] = img[y][x][1]
                                dst[y][x][2] = img[y][x][2]
                    cv2.imshow(window_name, dst) 
                #右クリックがあったら終了
                elif mouseData.getEvent() == cv2.EVENT_RBUTTONDOWN:
                    break;
                        
            cv2.destroyAllWindows()               
            cv2.imwrite(name0, dst)
#            abs_name = data_dir_path1 + '/' + file_name
#            img = cv2.imread(abs_name)
#            print(abs_name)
#
#            height, width,channels = img.shape       
#            height_split = 5
#            width_split = 7
#            new_img_height = int(height / height_split)
#            new_img_width = int(width / width_split)
#            num  = 0
#            for h in range(height_split):
#                height_start = h * new_img_height
#                height_end = height_start + new_img_height
#                
#                for w in range(width_split):
#                    width_start = w * new_img_width
#                    width_end = width_start + new_img_width
#                    num = num +1
#                    name0 = "./gt_image/"+file_name
#
#                    clp1 = img[height_start:height_end, width_start:width_end]
#                    size = (size_new*5, size_new*5)
#                    resize_clp1 = cv2.resize(clp1,size)
#
#                    cv2.imshow('image', resize_clp1)
#                    key = cv2.waitKey(0)&0xff
#                    if key == ord('q'):
#                        for y  in range(height_start-1,height_end-1):
#                            for x  in range(width_start-1,width_end-1):
#                                img[y][x][2] = 255  
#                    
#                    elif key == ord('p'):
#                        cv2.destroyAllWindows()
#                        
#                    cv2.imwrite(name0, img)

