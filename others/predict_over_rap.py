# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 11:33:03 2018

@author: g089v
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:55:20 2018

@author: g089v
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 11:41:29 2018

@author: g089v
"""

# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
from chainer import serializers
from simple_convnet4 import CNN
#from simple_convnet3 import CNN

import chainer.links as L
import cv2
from PIL import Image
from F1measure import F1_measure
import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.functions as F

def predict(filter_num = 5,inpaint = 1,save_file = ""):


    filter_str = str(filter_num) 

    seg = 0

    model =  L.Classifier(CNN())
#    serializers.load_npz("./CNN_modelmedian.npz", model)

#    serializers.load_npz("./snap_shot/old_medi41/CNN_modelmedian"+(filter_str)+".npz", model)
#    name, ext = os.path.splitext(os.path.basename(args.model))

    serializers.load_npz("./snap_shot/medi41/median41_"+(filter_str)+"snapshot_epoch-50", model, path= 'updater/model:main/')    
    val_num = 0.6

    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    data_channels = 1
    data_dir_path1 = u"./data/2.5m_median41"
    data_dir_path2 = u"./data/2.5m_half"
    file_list = os.listdir(r'./data/2.5m_half/')
#    data_dir_path1 = u"./data/half_data"
#    data_dir_path2 = u"./data/half_data"    
#    file_list = os.listdir(r"../95-5/"+(filter_str)+"/train/")
    nnum = 0
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            nnum = nnum + 1
            print(file_name)
            abs_name1 = data_dir_path1 + '/' + file_name
            abs_name2 = data_dir_path2 + '/' + file_name
            file_name = file_name[:-4]
            
            if data_channels == 3 or data_channels == 33 :
                src_img = cv2.imread(abs_name1)
                if inpaint == 1:
                    src_img = cv2.imread("./data/inpaint_area/" + file_name+".bmp")
                height, width,channela = src_img.shape
            
            if data_channels == 1 or data_channels == 13:
                src_img = cv2.imread(abs_name1,0)
                
                if inpaint == 1:
                    src_img = cv2.imread("./data/2.5m_inpaint/" + file_name+".bmp",0)
                    print("read inpaint" )
                height, width = src_img.shape    
                
            if data_channels == 21:
                src_img41 = cv2.imread(abs_name1,0)
                src_img21 = cv2.imread("./data/median21/"+file_name+".bmp",0)
                
                if inpaint == 1:
                    src_img41 = cv2.imread("./data/inpaint_median41/" + file_name+".bmp",0)
                    src_img21 = cv2.imread("./data/inpaint_median21/" + file_name+".bmp",0)

                    print("read inpaint" )
#                src_img51 = cv2.imread("./data/nlm51/"+file_name+".bmp",0)
                height, width = src_img.shape
#                cv2.imshow("a",src_img)
#                cv2.imshow("b",src_img31)
#            cv2.imshow("c",src_img)
#            cv2.waitKey(0)                
                
            dst_img = cv2.imread(abs_name2)            
            f1_img = cv2.imread(abs_name2)
           
            
#           

            

            
            mask  = np.zeros((height, width), np.uint8)
#            cv2.imshow("a",mask)
#            cv2.waitKey(0)            
            over_rap = 25
            new_img_height = 50
            new_img_width = 50
            width_split = int(width/(new_img_width-over_rap))-1
            height_split = int(height/(new_img_height-over_rap))-1

            a1,b1,c1 = 0,0,0           
            num  = 0
            for h in range(height_split):
                height_start = h * over_rap
                height_end = height_start + new_img_height
        
                for w in range(width_split):
                    
                    width_start = w * over_rap
                    width_end = width_start + new_img_width
#                    print(-height_start+height_end)
#                    print(-width_start+width_end)
#                    print("\n\n")
                    num = num +1

                    clp1 = src_img[height_start:height_end, width_start:width_end]
                    PIL_data=Image.fromarray(clp1)
#                    im_list = np.asarray(clp1)
#                    #貼り付け
#                    plt.imshow(im_list)
#                    #表示
#                    plt.show()

                    if data_channels == 3:
                        
                        r,g,b = PIL_data.split()
                        rImgData = np.asarray(np.float32(r)/255.0)
                        gImgData = np.asarray(np.float32(g)/255.0)
                        bImgData = np.asarray(np.float32(b)/255.0)
                        imgData = np.asarray([rImgData, gImgData, bImgData])
    #                    grayImgData = np.asarray(np.float32(PIL_data)/255.0)
    
                        x = imgData





                    if data_channels == 33:
                        
                        r,g,b = PIL_data.split()
                        rImgData = np.asarray(np.float32(r)/255.0)
                        gImgData = np.asarray(np.float32(g)/255.0)
                        bImgData = np.asarray(np.float32(b)/255.0)
                        
                        seg_n = "seg"
                        if os.path.isfile("./data/"+seg_n+"_hall_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img1 = np.array(Image.open("./data/"+seg_n+"_hall_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            a1 = a1 + 1
                        else :
                            seg_img1 = np.array(np.full((50,50), 255, dtype=np.uint8))
            
                        if os.path.isfile("./data/"+seg_n+"_shadow_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img2 = np.array(Image.open("./data/"+seg_n+"_shadow_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            b1 = b1 + 1
                        else :
                            seg_img2 = np.array(np.full((50,50), 255, dtype=np.uint8))
                            
                        if os.path.isfile("./data/"+seg_n+"_hyouzi_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img3 = np.array(Image.open("./data/"+seg_n+"_hyouzi_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            c1 = c1 + 1
                        else :
                            seg_img3 = np.array(np.full((50,50), 255, dtype=np.uint8))
#          
                        seg1  = np.asarray(np.float32(seg_img1)/255.0)
                        seg2  = np.asarray(np.float32(seg_img2)/255.0)
                        seg3  = np.asarray(np.float32(seg_img3)/255.0)
                        
                        imgData = np.asarray([bImgData, gImgData, rImgData,seg1,seg2,seg3])
                        x = imgData



                    
                    if data_channels == 1:
                        grayImgData = np.asarray(np.float32(PIL_data)/255.0)
                        x = grayImgData[None,...]
                    
                    
                    if data_channels == 13: #領域分割結果を合わせて入力

            #            print(pathAndLabel[0])
                        grayImgData = np.asarray(np.float32(PIL_data)/255.0)
                        

                        if os.path.isfile("./data/2.5m_hall_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img1 = np.array(Image.open("./data/2.5m_hall_hall/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            a1 = a1 + 1
                        else :
                            seg_img1 = np.array(np.full((50,50), 0, dtype=np.uint8))
            
                        if os.path.isfile("./data/2.5m_shadow_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img2 = np.array(Image.open("./data/2.5m_shadow_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            b1 = b1 + 1
                        else :
                            seg_img2 = np.array(np.full((50,50), 0, dtype=np.uint8))
                            
                        if os.path.isfile("./data/2.5m_hyouzi_batch/"+file_name+"_"+str(num)+".bmp") == True:
                            seg_img3 = np.array(Image.open("./data/2.5m_hyouzi_batch/"+file_name+"_"+str(num)+".bmp").convert('L'))
                            c1 = c1 + 1
                        else :
                            seg_img3 = np.array(np.full((50,50), 0, dtype=np.uint8))
#          
                        seg1  = np.asarray(np.float32(seg_img1)/255.0)
                        seg2  = np.asarray(np.float32(seg_img2)/255.0)
                        seg3  = np.asarray(np.float32(seg_img3)/255.0)
                        
                        
                        imgData = np.asarray([grayImgData,seg1,seg2,seg3])
                        x = imgData
                    if data_channels == 21:
                        clp41 = src_img41[height_start:height_end, width_start:width_end]
                        PIL_data41=Image.fromarray(clp41)
                        
                        clp21 = src_img21[height_start:height_end, width_start:width_end]
                        PIL_data21=Image.fromarray(clp21)
                        
                        grayImgData41 = np.asarray(np.float32(PIL_data41)/255.0)
                        grayImgData21 = np.asarray(np.float32(PIL_data21)/255.0)                        
                        imgData = np.asarray([grayImgData41,grayImgData21])
                        x = imgData
                    
                    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                        y = model.predictor(x[None,...]).data.argmax(axis=1)[0]
                        yy = model.predictor(x[None,...])
                        rate = F.softmax(yy.data)[0][1]

#                        print(rate.data*100)
#                    if rate.data >= 0.2:
                    if y == 1:
#                        print('CNN:  crack:')
#                        plt.imshow(grayImgData, cmap='gray')
#                        plt.show()
                        for y  in range(height_start,height_end):
                            for x  in range(width_start,width_end):
                               mask[y][x] = 255
                               dst_img[y][x][2] = dst_img[y][x][2] + 20
                               if dst_img[y][x][2] >=255:
                                   dst_img[y][x][2] = 254
                                   
            print(a1,b1,c1)
#            a,b,c,d = F1_measure(f1_img,mask,file_name,seg,"./data/t_gt_gray_own/")
                               
#            a,b,c,d = F1_measure(f1_img,mask,file_name,seg,"./data/2.5m_gt_gray_own/","../95-5/"+filter_str+"/"+save_file)
#            TP = TP + a
#            FP = FP + b
#            FN = FN + c
#            TN = TN + d

        
            cv2.imwrite('CNN_output/'+file_name+'.bmp', dst_img)     
#    Precision = (TP+0.001)/(TP+FP+0.001)
#    Recall = (TP+0.001)/(TP+FN+0.001)
#    F1 = 2*Recall*Precision/(Recall+Precision)
#    Specificity = (TN+0.001)/(TN+FP+0.001)      
#
#    print ("Precision={:.4}".format(Precision))
#    print ("Recall={:.4}".format(Recall))
#    print ("Specificity={:.4}".format(Specificity)) 
#    print ("F1={:.4}\n\n".format(F1))
##    f = open("./F1/F1.txt",'w')    
#    f = open("./../95-5/"+filter_str+"/"+save_file+"/F1.txt",'w')
#    f.write("Precision={:.4}".format(Precision)+'\n')
#    f.write("Recall={:.4}".format(Recall)+"\n")
#    f.write("F1={:.4}".format(F1)+'\n')
#    f.write("Specificity={:.4}".format(Specificity)+'\n')
#    f.close() # ファイルを閉じる
#    

    filter_num = filter_num + 1      
    return 0    

if __name__ == '__main__':
    for n in range(5):
        print("predict"+str(n+1))
        predict(n+1,0,"rgb_seg_2.5m/")

