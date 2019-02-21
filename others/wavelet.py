# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:01:39 2017

@author: g089v
"""

import pywt
import numpy as np
import cv2
import os

def image_normalization(src_img):
    """
    白飛び防止のための正規化処理
    cv2.imshowでwavelet変換された画像を表示するときに必要（大きい値を持つ画像の時だけ）
    """
    norm_img = (src_img - np.min(src_img)) / (np.max(src_img) - np.min(src_img))
    return norm_img

def merge_images(file_name,cA, cH_V_D):
    """numpy.array を４つ(左上、(右上、左下、右下))連結させる"""
    cH, cV, cD = cH_V_D
    cH = image_normalization(cH) # 外してもok
    cV = image_normalization(cV) # 外してもok
    cD = image_normalization(cD) # 外してもok
    cA = cA[0:cH.shape[0], 0:cV.shape[1]] # 元画像が2の累乗でない場合、端数ができることがあるので、サイズを合わせる。小さい方に合わせます。
    
#    cA = cA*255
#    cH = cH*255
#    cV = cV*255
#    cD = cD*255
    cv2.imwrite("./cH/"+file_name, cH)
    cv2.imwrite("./cV/"+file_name, cV)
    cv2.imwrite("./cD/"+file_name, cD)

#    cv2.imshow('cV', cV)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

    return np.vstack((np.hstack((cA,cH)), np.hstack((cV, cD)))) # 左上、右上、左下、右下、で画素をくっつける

def coeffs_visualization(file_name,cof):
    norm_cof0 = cof[0]
    norm_cof0 = image_normalization(norm_cof0) # 外してもok
    merge = norm_cof0

    for i in range(1, len(cof)):
        merge = merge_images(file_name,merge, cof[i])  # ４つの画像を合わせていく
     
    #cv2.imwrite("./"+file_name, merge)
#    cv2.imshow('', merge)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()

def wavelet_transform_for_image(src_image, level, M_WAVELET="db1", mode="sym"):
    data = src_image.astype(np.float64)
    coeffs = pywt.wavedec2(data, M_WAVELET, level=level, mode=mode)
    return coeffs

if __name__ == "__main__":
    data_dir_path = u"./grass/"
    file_list = os.listdir(r'./grass/')

    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            print('ファイル名:',abs_name)
            im = cv2.imread(abs_name)
#            cv2.imshow('', im)
#            cv2.waitKey(0)
            cv2.destroyAllWindows()
            orgHeight, orgWidth = im.shape[:2]
            size = (700, 1000)
            im = cv2.resize(im, size)

            LEVEL = 1
            MOTHER_WAVELET = "db1"            
            print('LEVEL :', LEVEL)
            print('MOTHER_WAVELET', MOTHER_WAVELET)
            #print('original image size: ', im.shape)
        
            """
            各BGRチャネル毎に変換
            cv2.imreadはB,G,Rの順番で画像を吐き出すので注意
            """
            B = 0
            G = 1
            R = 2
            coeffs_B = wavelet_transform_for_image(im[:, :, B], LEVEL, M_WAVELET=MOTHER_WAVELET)
            coeffs_G = wavelet_transform_for_image(im[:, :, G], LEVEL, M_WAVELET=MOTHER_WAVELET)
            coeffs_R = wavelet_transform_for_image(im[:, :, R], LEVEL, M_WAVELET=MOTHER_WAVELET)
        
            coeffs_visualization(file_name,coeffs_B)
#            coeffs_visualization(coeffs_G)
#            coeffs_visualization(coeffs_R)

