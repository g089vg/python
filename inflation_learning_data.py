

# -*- coding: utf-8 -*-
"""
Created on Fri May 25 14:33:58 2018

@author: g089v
"""


import numpy as np
import cv2
import os
import random
name = 1
name = (str)(name)


def addGaussianNoise(src):
    row,col,ch= src.shape
    mean = 0
    var = 0.1
    sigma = 20
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = src + gauss

    return noisy

def Delete_File (top = '削除したいディレクトリ'):     
  for root, dirs, files in os.walk(top, topdown=False):
      for name in files:
          os.remove(os.path.join(root, name))
      for name in dirs:
          os.rmdir(os.path.join(root, name))
          
def Translation(img,num):
    if num == 0:
        return img
    h, w = img.shape[0],img.shape[1]
#    size = (w, h)
    # 回転角の指定
    angle = num
    angle_rad = angle/180.0*np.pi
    
    # 回転後の画像サイズを計算
    w_rot = int(np.round(h*np.absolute(np.sin(angle_rad))+w*np.absolute(np.cos(angle_rad))))
    h_rot = int(np.round(h*np.absolute(np.cos(angle_rad))+w*np.absolute(np.sin(angle_rad))))
    size_rot = (w_rot, h_rot)
    
    # 元画像の中心を軸に回転する
    center = (w/2, h/2)
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 平行移動を加える (rotation + translation)
    if num == 90:
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2
        affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2-1
    if num == 180:
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2-1
        affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2-1    
        
    if num == 270:
        affine_matrix = rotation_matrix.copy()
        affine_matrix[0][2] = affine_matrix[0][2] -w/2 + w_rot/2-1
        affine_matrix[1][2] = affine_matrix[1][2] -h/2 + h_rot/2
        
    img_rot = cv2.warpAffine(img, affine_matrix, size_rot, flags=cv2.INTER_CUBIC)    
    if num != 0 :
        return(img_rot)


def split_stack(src):
    v_size = 50
    h_size = 50
    img = src[:v_size, :h_size]

    v_split = 2
    h_split = 2
    
    out_img = []
    [out_img.extend(np.hsplit(h_img, h_split))
      for h_img in np.vsplit(img, v_split)]  
    
    l = [0, 90, 180, 270]
    for i in range(4):
 
        out_img[i] = Translation(out_img[i],random.choice(l))
        

    shuffle = np.random.permutation(out_img)

    img = np.vstack((
        np.hstack(shuffle[0:2]),
        np.hstack(shuffle[2:4]),
    ))
    return(img)
    
    
min_table = 50
max_table = 205
diff_table = max_table - min_table
LUT_HC = np.arange(256, dtype = 'uint8' )
# ハイコントラストLUT作成
for i in range(0, min_table):
    LUT_HC[i] = 0
for i in range(min_table, max_table):
    LUT_HC[i] = 255 * (i - min_table) / diff_table
for i in range(max_table, 255):
    LUT_HC[i] = 255
LUT_G1 = np.arange(256, dtype = 'uint8' )
LUT_G2 = np.arange(256, dtype = 'uint8' )
gamma1 = 0.8
gamma2 = 0.9
for i in range(256):
    LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
    LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)    




crack0 = 0
crack1 = 0


num = 0


file_list = os.listdir(r"./data/half_data/")
mode = "rgb"
channels = 3
for file_name in file_list:
    root, ext = os.path.splitext(file_name)
    if ext == u'.bmp':
        file_name =  file_name[:-4]
        print(file_name)
        
        Delete_File("./data/Ground_Truth/"+file_name+"/0/inf_"+mode+"/")
        Delete_File("./data/Ground_Truth/"+file_name+"/1/inf_"+mode+"/")
        
        file_list0 = os.listdir(r"./data/Ground_Truth/"+file_name+"/0/"+mode+"/")
        file_list1 = os.listdir(r"./data/Ground_Truth/"+file_name+"/1/"+mode+"/")       
        data_dir_path0 = u"./data/Ground_Truth/"+file_name+"/0/"+mode+"/"
        data_dir_path1 = u"./data/Ground_Truth/"+file_name+"/1/"+mode+"/"
        outputname0 = "./data/Ground_Truth/"+file_name+"/0/inf_"+mode+"/"
        outputname1 = "./data/Ground_Truth/"+file_name+"/1/inf_"+mode+"/"
        for xx in range(2):
            file_list2 = []
            data_dir_path2 = []
            outputname2 = []
            
            if xx == 0:
                crack0=crack0+1
                file_list2 = file_list0
                data_dir_path2 = data_dir_path0
                outputname2 = outputname0
            if xx == 1:
#                print("no crack")
                crack1=crack1+1
                file_list2 = file_list1
                data_dir_path2 = data_dir_path1
                outputname2 = outputname1
                
            for file_name2 in file_list2:
                root, ext = os.path.splitext(file_name2)
                if ext == u'.bmp':
                    print("1----"+data_dir_path2+file_name2)  

                    img = cv2.imread(data_dir_path2+file_name2)
                    invgray = cv2.bitwise_not(img)
#                    cv2.imshow("data",img)
#                    cv2.waitKey(10)
                    file_name2 =  file_name2[:-4]       
                    
                    cv2.imwrite(outputname2+file_name2+"_s1.bmp",split_stack(img))
                    cv2.imwrite(outputname2+file_name2+"_s2.bmp",split_stack(img))
                    cv2.imwrite(outputname2+file_name2+"_s3.bmp",split_stack(img))
                    cv2.imwrite(outputname2+file_name2+"_s4.bmp",split_stack(img)) 
                            
                    cv2.imwrite(outputname2+file_name2+"_0_0.bmp",img)                    
                    cv2.imwrite(outputname2+file_name2+"_0_90.bmp",Translation(img,90))
                    cv2.imwrite(outputname2+file_name2+"_0_180.bmp",Translation(img,180))
                    cv2.imwrite(outputname2+file_name2+"_0_027.bmp",Translation(img,270))
                    img_flip = cv2.flip(img, 1)
                    cv2.imwrite(outputname2+file_name2+"_1_0.bmp",img_flip)
                    cv2.imwrite(outputname2+file_name2+"_1_90.bmp",Translation(img_flip,90))
                    cv2.imwrite(outputname2+file_name2+"_1_180.bmp",Translation(img_flip,180))
                    cv2.imwrite(outputname2+file_name2+"_1_270.bmp",Translation(img_flip,270))    
            #                    12
                    
                    high_cont_img = cv2.LUT(img, LUT_HC)     
                    cv2.imwrite(outputname2+file_name2+"_high_cont_s1.bmp",split_stack(high_cont_img))
                    cv2.imwrite(outputname2+file_name2+"_high_cont_s2.bmp",split_stack(high_cont_img))
                    cv2.imwrite(outputname2+file_name2+"_high_cont_s3.bmp",split_stack(high_cont_img))
                    cv2.imwrite(outputname2+file_name2+"_high_cont_s4.bmp",split_stack(high_cont_img)) 
                    
                    cv2.imwrite(outputname2+file_name2+"_high_cont_0_0.bmp",high_cont_img)
                    cv2.imwrite(outputname2+file_name2+"_high_cont_0_90.bmp",Translation(high_cont_img,90))
                    cv2.imwrite(outputname2+file_name2+"_high_cont_0_180.bmp",Translation(high_cont_img,180))
                    cv2.imwrite(outputname2+file_name2+"_high_cont_0_270.bmp",Translation(high_cont_img,270))  
                    high_cont_flip = cv2.flip(high_cont_img, 1)
                    cv2.imwrite(outputname2+file_name2+"_high_cont_1_0.bmp",high_cont_flip)
                    cv2.imwrite(outputname2+file_name2+"_high_cont_1_90.bmp",Translation(high_cont_flip,90))
                    cv2.imwrite(outputname2+file_name2+"_high_cont_1_180.bmp",Translation(high_cont_flip,180))
                    cv2.imwrite(outputname2+file_name2+"_high_cont_1_270.bmp",Translation(high_cont_flip,270))    
            #                   24
                    
                    gamma_img1 = cv2.LUT(img, LUT_G1)   
                    cv2.imwrite(outputname2+file_name2+"_gamma1_s1.bmp",split_stack(gamma_img1))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_s2.bmp",split_stack(gamma_img1))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_s3.bmp",split_stack(gamma_img1))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_s4.bmp",split_stack(gamma_img1)) 
                    
                    cv2.imwrite(outputname2+file_name2+"_gamma1_0_0.bmp",gamma_img1)
                    cv2.imwrite(outputname2+file_name2+"_gamma1_0_90.bmp",Translation(gamma_img1,90))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_0_180.bmp",Translation(gamma_img1,180))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_0_270.bmp",Translation(gamma_img1,270))    
                    gamma_img1_flip = cv2.flip(gamma_img1, 1)
                    cv2.imwrite(outputname2+file_name2+"_gamma1_1_0.bmp",gamma_img1_flip)
                    cv2.imwrite(outputname2+file_name2+"_gamma1_1_90.bmp",Translation(gamma_img1_flip,90))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_1_180.bmp",Translation(gamma_img1_flip,180))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_1_270.bmp",Translation(gamma_img1_flip,270))                     
            #                    36
                    
                    gamma_img2 = cv2.LUT(img, LUT_G2)  
                    cv2.imwrite(outputname2+file_name2+"_gamma2_s1.bmp",split_stack(gamma_img2))
                    cv2.imwrite(outputname2+file_name2+"_gamma2_s2.bmp",split_stack(gamma_img2))
                    cv2.imwrite(outputname2+file_name2+"_gamma2_s3.bmp",split_stack(gamma_img2))
                    cv2.imwrite(outputname2+file_name2+"_gamma2_s4.bmp",split_stack(gamma_img2))  
                    
                    cv2.imwrite(outputname2+file_name2+"_gamma2_0_0.bmp",gamma_img2)
                    cv2.imwrite(outputname2+file_name2+"_gamma2_0_90.bmp",Translation(gamma_img2,90))
                    cv2.imwrite(outputname2+file_name2+"_gamma2_0_180.bmp",Translation(gamma_img2,180))
                    cv2.imwrite(outputname2+file_name2+"_gamma2_0_270.bmp",Translation(gamma_img2,270))    
                    gamma_img2_flip = cv2.flip(high_cont_img, 1)
                    cv2.imwrite(outputname2+file_name2+"_gamma2_1_0.bmp",gamma_img2_flip)
                    cv2.imwrite(outputname2+file_name2+"_gamma2_1_90.bmp",Translation(gamma_img2_flip,90))
                    cv2.imwrite(outputname2+file_name2+"_gamma2_1_180.bmp",Translation(gamma_img2_flip,180))
                    cv2.imwrite(outputname2+file_name2+"_gamma2_1_270.bmp",Translation(gamma_img2_flip,270))                     
            #                   48
                    
                    inv = cv2.bitwise_not(img)
                    cv2.imwrite(outputname2+file_name2+"_inv_s1.bmp",split_stack(inv))
                    cv2.imwrite(outputname2+file_name2+"_inv_s2.bmp",split_stack(inv))
                    cv2.imwrite(outputname2+file_name2+"_inv_s3.bmp",split_stack(inv))
                    cv2.imwrite(outputname2+file_name2+"_inv_s4.bmp",split_stack(inv))       
                    
                    cv2.imwrite(outputname2+file_name2+"_inv_0_0.bmp",inv)
                    cv2.imwrite(outputname2+file_name2+"_inv_0_90.bmp",Translation(inv,90))
                    cv2.imwrite(outputname2+file_name2+"_inv_0_180.bmp",Translation(inv,180))
                    cv2.imwrite(outputname2+file_name2+"_inv_0_270.bmp",Translation(inv,270))    
                    inv_flip = cv2.flip(inv, 1)
                    cv2.imwrite(outputname2+file_name2+"_inv_1_0.bmp",inv_flip)
                    cv2.imwrite(outputname2+file_name2+"_inv_1_90.bmp",Translation(inv_flip,90))
                    cv2.imwrite(outputname2+file_name2+"_inv_1_180.bmp",Translation(inv_flip,180))
                    cv2.imwrite(outputname2+file_name2+"_inv_1_270.bmp",Translation(inv_flip,270))  
            #                   60                    
                    
                    gamma1_hc = cv2.LUT(high_cont_img, LUT_G1) 
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_s1.bmp",split_stack(gamma1_hc))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_s2.bmp",split_stack(gamma1_hc))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_s3.bmp",split_stack(gamma1_hc))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_s4.bmp",split_stack(gamma1_hc))
                    
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_0_0.bmp",gamma1_hc)
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_0_90.bmp",Translation(gamma1_hc,90))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_0_180.bmp",Translation(gamma1_hc,180))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_0_270.bmp",Translation(gamma1_hc,270))    
                    gamma1_hc_flip = cv2.flip(gamma1_hc, 1)
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_1_0.bmp",gamma1_hc_flip)
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_1_90.bmp",Translation(gamma1_hc_flip,90))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_1_180.bmp",Translation(gamma1_hc_flip,180))
                    cv2.imwrite(outputname2+file_name2+"_gamma1_hc_1_270.bmp",Translation(gamma1_hc_flip,270))   
            #                   72
                    if xx == 1:
                        gamma2_hc = cv2.LUT(high_cont_img, LUT_G1) 
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_s1.bmp",split_stack(gamma2_hc))
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_s2.bmp",split_stack(gamma2_hc))
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_s3.bmp",split_stack(gamma2_hc))
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_s4.bmp",split_stack(gamma2_hc))
                        
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_0_0.bmp",gamma2_hc)
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_0_90.bmp",Translation(gamma2_hc,90))
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_0_180.bmp",Translation(gamma2_hc,180))
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_0_270.bmp",Translation(gamma2_hc,270))    
                        gamma2_hc_flip = cv2.flip(gamma2_hc, 1)
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_1_0.bmp",gamma2_hc_flip)
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_1_90.bmp",Translation(gamma2_hc_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_1_180.bmp",Translation(gamma2_hc_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_gamma2_hc_1_270.bmp",Translation(gamma2_hc_flip,270))   
                #                   84
                
                        gussian = cv2.GaussianBlur(img,(3,3),0)
                        cv2.imwrite(outputname2+file_name2+"_gussian_s1.bmp",split_stack(gussian))
                        cv2.imwrite(outputname2+file_name2+"_gussian_s2.bmp",split_stack(gussian))
                        cv2.imwrite(outputname2+file_name2+"_gussian_s3.bmp",split_stack(gussian))
                        cv2.imwrite(outputname2+file_name2+"_gussian_s4.bmp",split_stack(gussian))
                        
                        cv2.imwrite(outputname2+file_name2+"_gussian_0_0.bmp",gussian)
                        cv2.imwrite(outputname2+file_name2+"_gussian_0_90.bmp",Translation(gussian,90))
                        cv2.imwrite(outputname2+file_name2+"_gussian_0_180.bmp",Translation(gussian,180))
                        cv2.imwrite(outputname2+file_name2+"_gussian_0_270.bmp",Translation(gussian,270))    
                        gussian_flip = cv2.flip(gussian, 1)
                        cv2.imwrite(outputname2+file_name2+"_gussian_1_0.bmp",gussian_flip)
                        cv2.imwrite(outputname2+file_name2+"_gussian_1_90.bmp",Translation(gussian_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_gussian_1_180.bmp",Translation(gussian_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_gussian_1_270.bmp",Translation(gussian_flip,270))
                        
                    #           96
                    
                    
                    
                        gussian_gamma1 = cv2.GaussianBlur(gamma_img1,(3,3),0)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_s1.bmp",split_stack(gussian_gamma1))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_s2.bmp",split_stack(gussian_gamma1))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_s3.bmp",split_stack(gussian_gamma1))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_s4.bmp",split_stack(gussian_gamma1))
                
                        
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_0_0.bmp",gussian_gamma1)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_0_90.bmp",Translation(gussian_gamma1,90))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_0_180.bmp",Translation(gussian_gamma1,180))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_0_270.bmp",Translation(gussian_gamma1,270))    
                        gussian_gamma1_flip = cv2.flip(gussian_gamma1, 1)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_1_0.bmp",gussian_gamma1_flip)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_1_90.bmp",Translation(gussian_gamma1_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_1_180.bmp",Translation(gussian_gamma1_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma1_1_270.bmp",Translation(gussian_gamma1_flip,270))
                #               108
                    
                        gussian_gamma2 = cv2.GaussianBlur(gamma_img2,(3,3),0)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_s1.bmp",split_stack(gussian_gamma2))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_s2.bmp",split_stack(gussian_gamma2))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_s3.bmp",split_stack(gussian_gamma2))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_s4.bmp",split_stack(gussian_gamma2))
                        
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_0_0.bmp",gussian_gamma2)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_0_90.bmp",Translation(gussian_gamma2,90))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_0_180.bmp",Translation(gussian_gamma2,180))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_0_270.bmp",Translation(gussian_gamma2,270))    
                        gussian_gamma2_flip = cv2.flip(gussian_gamma2, 1)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_1_0.bmp",gussian_gamma2_flip)
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_1_90.bmp",Translation(gussian_gamma2_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_1_180.bmp",Translation(gussian_gamma2_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_gussian_gamma2_1_270.bmp",Translation(gussian_gamma2_flip,270))
                #               120
                        
                        noise = addGaussianNoise(img)
                        cv2.imwrite(outputname2+file_name2+"_noise_s1.bmp",split_stack(noise))
                        cv2.imwrite(outputname2+file_name2+"_noise_s2.bmp",split_stack(noise))
                        cv2.imwrite(outputname2+file_name2+"_noise_s3.bmp",split_stack(noise))
                        cv2.imwrite(outputname2+file_name2+"_noise_s4.bmp",split_stack(noise))
                        
                        
                        cv2.imwrite(outputname2+file_name2+"_noise_0_0.bmp",noise)
                        cv2.imwrite(outputname2+file_name2+"_noise_0_90.bmp",Translation(noise,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_0_180.bmp",Translation(noise,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_0_270.bmp",Translation(noise,270))    
                        noise_flip = cv2.flip(noise, 1)
                        cv2.imwrite(outputname2+file_name2+"_noise_1_0.bmp",noise_flip)
                        cv2.imwrite(outputname2+file_name2+"_noise_1_90.bmp",Translation(noise_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_1_180.bmp",Translation(noise_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_1_270.bmp",Translation(noise_flip,270))        
                #               132
                        
                        noise_gamma1 = addGaussianNoise(gamma_img1)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_s1.bmp",split_stack(noise_gamma1))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_s2.bmp",split_stack(noise_gamma1))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_s3.bmp",split_stack(noise_gamma1))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_s4.bmp",split_stack(noise_gamma1))
                        
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_0_0.bmp",noise_gamma1)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_0_90.bmp",Translation(noise_gamma1,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_0_180.bmp",Translation(noise_gamma1,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_0_270.bmp",Translation(noise_gamma1,270))    
                        noise_gamma1_flip = cv2.flip(noise_gamma1, 1)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_1_0.bmp",noise_gamma1_flip)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_1_90.bmp",Translation(noise_gamma1_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_1_180.bmp",Translation(noise_gamma1_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma1_1_270.bmp",Translation(noise_gamma1_flip,270))
                        
                #               144     
                        
                        noise_gamma2 = addGaussianNoise(gamma_img2)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_s1.bmp",split_stack(noise_gamma2))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_s2.bmp",split_stack(noise_gamma2))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_s3.bmp",split_stack(noise_gamma2))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_s4.bmp",split_stack(noise_gamma2))
                        
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_0_0.bmp",noise_gamma1)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_0_90.bmp",Translation(noise_gamma1,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_0_180.bmp",Translation(noise_gamma1,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_0_270.bmp",Translation(noise_gamma1,270))    
                        noise_gamma2_flip = cv2.flip(noise_gamma2, 1)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_1_0.bmp",noise_gamma2_flip)
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_1_90.bmp",Translation(noise_gamma2_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_1_180.bmp",Translation(noise_gamma2_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_gamma2_1_270.bmp",Translation(noise_gamma2_flip,270))       
                        
                 #               156    
                 
                        noise_hc = addGaussianNoise(high_cont_img)
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_s1.bmp",split_stack(noise_hc))
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_s2.bmp",split_stack(noise_hc))
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_s3.bmp",split_stack(noise_hc))
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_s4.bmp",split_stack(noise_hc))
                        
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_0_0.bmp",noise_hc)
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_0_90.bmp",Translation(noise_hc,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_0_180.bmp",Translation(noise_hc,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_0_270.bmp",Translation(noise_hc,270))    
                        noise_hc_flip = cv2.flip(noise_hc, 1)
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_1_0.bmp",noise_hc_flip)
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_1_90.bmp",Translation(noise_hc_flip,90))
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_1_180.bmp",Translation(noise_hc_flip,180))
                        cv2.imwrite(outputname2+file_name2+"_noise_hc_1_270.bmp",Translation(noise_hc_flip,270))   
