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
    data_dir_path1 = u"./data/2.5m_half/"


#    data_dir_path2 = u"./gt_half"
    file_list = os.listdir(r'./data/2.5m_half/')
            
    read_num = 0
        
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        read_num = read_num + 1
        if ext == u'.bmp' and file_name == "03_2017042511071247.bmp":
            name1 = "./data/2.5m_gt_gray_own3/"+file_name
#            name1 = "./data/t_gt_gray_own/"+file_name
            abs_name = data_dir_path1 + file_name
            img = cv2.imread(abs_name)
            img2 = cv2.imread(abs_name)
            dst = cv2.imread(abs_name)
            


            height, width,channels = img.shape       
            mask  = np.zeros((height, width), np.uint8)
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
                    for y  in range(height):
                        for x  in range(width):
                            
                            if p_label[x][y] == label and gt_label[x][y] == 0:
                                gt_label[x][y] = 1
                                
                            elif p_label[x][y] == label and gt_label[x][y] == 1:
                                gt_label[x][y] = 0                                
                            
                    for y  in range(height):
                        for x  in range(width):
                            
                            if gt_label[x][y] == 1:
                                dst[y][x][0] = dst[y][x][0]
                                dst[y][x][1] = dst[y][x][1]
                                dst[y][x][2] = 255
                                
                                img2[y][x][0] = img[y][x][0]
                                img2[y][x][1] = img[y][x][1]
                                img2[y][x][2] = 255                                
                                mask[y][x] = 255
                            if gt_label[x][y] == 0:
                                dst[y][x][0] = img[y][x][0]
                                dst[y][x][1] = img[y][x][1]
                                dst[y][x][2] = img[y][x][2]
                                
                                img2[y][x][0] = img[y][x][0]
                                img2[y][x][1] = img[y][x][1]
                                img2[y][x][2] = img[y][x][2]  
                                mask[y][x] = 0
                    cv2.imshow(window_name, dst) 
                #中クリックがあったら終了
                elif mouseData.getEvent() == cv2.EVENT_MBUTTONDOWN:
                    break;
                        
            cv2.destroyAllWindows()               
            cv2.imwrite(name1, mask)
