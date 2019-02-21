# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:29:31 2018

@author: g089v
"""
import cv2
import glob

fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('video.mp4', fourcc, 5.0, (800, 600))

folderName = "road_1"
#画像ファイルの一覧を取得
picList = glob.glob(folderName + "\*.jpg")
     

for i in range(len(picList)):
    img = cv2.imread(picList[i])
    img = cv2.resize(img, (800, 600))
    video.write(img)

video.release()
