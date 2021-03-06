# -*- coding: utf-8 -*-
"""
Created on Mon May 28 09:54:18 2018

@author: g089v
"""
from chainer.datasets import tuple_dataset
from PIL import Image
import numpy as np
import glob
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def Load_Dataset(Folder_Name, mode,channels, num,gen = 0):


    allData = []
    a = 0
    crack_0 = 0
    crack_1 = 0
    file_list = os.listdir(Folder_Name) 
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            file_name =  file_name[:-4]

            
            label0 ="./data/Ground_Truth/"+file_name+"/0/inf_"+mode+"/"
            label1 = "./data/Ground_Truth/"+file_name+"/1/inf_"+mode+"/"
#            if channels == 5:  
#                label0 ="./data/Ground_Truth/"+file_name+"/0/"+mode+"/"
#                label1 = "./data/Ground_Truth/"+file_name+"/1/"+mode+"/"
            pathsAndLabels = []
            pathsAndLabels.append(np.asarray([label0, 0]))
            pathsAndLabels.append(np.asarray([label1, 1]))
    
            # データを混ぜて、trainとtestがちゃんとまばらになるように。
#    pathsAndLabels.append(np.asarray(["./data/gen_median41/", 1]))    

            for pathAndLabel in pathsAndLabels:
                path = pathAndLabel[0]
                label = pathAndLabel[1]
                imagelist = glob.glob(path + "*")

                for imgName in imagelist:
                    allData.append([imgName, label])
                    if label == str(1):
                        crack_1 = crack_1 +1
                    if label == str(0):
                        crack_0 = crack_0 +1
                    a = a +1
    normal_data = len(allData)   
    if gen == 1:
#merge segmantaion data                
        pathsAndLabels2 = []                   
        pathsAndLabels2.append(np.asarray(["./data/gen_median41_1/", 1]))    
        for pathAndLabel2 in pathsAndLabels2:
            path2 = pathAndLabel2[0]
            label2 = pathAndLabel2[1]
            imagelist2 = glob.glob(path2 + "*")
    
            for imgName2 in imagelist2:
                allData.append([imgName2, label2])
    
                
        generate_data= len(allData)-  normal_data          
        print("allData"+"="+str(normal_data)+"("+str(crack_0)+","+str(crack_1)+")+"+str(generate_data))   
        
    print("allData"+"="+str(normal_data)+"("+str(crack_0)+","+str(crack_1)+")"    )                              
    allData = np.random.permutation(allData)
    number = len(allData)  
    threshold = np.int32(len(allData) -1000)
    print("train="+str(threshold)+",test="+str(1000))
    load_num = 0
    load_rate = 0.0


    if channels == 0:
        print("Load numpy Data")
        imageData = np.load('./np_data/image_'+Folder_Name+mode+'.npy')
        labelData = np.load('./np_data/label_'+Folder_Name+mode+'.npy')
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        print(len(imageData))
        print(threshold)
        return train,test   



    if channels == 1: #チャンネル数が1つの場合
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
#            print(pathAndLabel[0])
            img = np.array(Image.open(pathAndLabel[0]).convert('L') )
    
            grayImgData = np.asarray(np.float32(img)/255.0)
            imgData = np.asarray([grayImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))
        np.save('./np_data/image_95-5_'+num+mode+'.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'.npy', labelData)
            
#        threshold = np.int32(len(imageData)/8*7)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        print("train:"+str(len(imageData)))
        print("value:"+str(len(imageData)-threshold))
        return train,test    

    if channels == 13: #領域分割結果を合わせて入力
        print("#領域分割結果を合わせて入力")
        imageData = []
        labelData = []
        a,b,c,d,e,f= 0,0,0,0,0,0
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
            GT = pathAndLabel[0][55:]
            GEN =  pathAndLabel[0][22:]
            
#            print( pathAndLabel[0])
#            print(GT)
#            print(GEN+"\n\n")
            img = np.array(Image.open(pathAndLabel[0]).convert('L') )

            if os.path.isfile("./data/segmantation_mask/seg_hall_inf/"+GT) == True:
                seg_img1 = np.array(Image.open("./data/segmantation_mask/seg_hall_inf/"+GT).convert('L'))
                a = a + 1
            else :
                seg_img1 = np.array(np.full((50,50), 255, dtype=np.uint8))

            if os.path.isfile("./data/segmantation_mask/seg_shadow_inf/"+GT) == True:
                seg_img2 = np.array(Image.open("./data/segmantation_mask/seg_shadow_inf/"+GT).convert('L'))
                b = b + 1
            else :
                seg_img2 = np.array(np.full((50,50), 255, dtype=np.uint8))
                
            if os.path.isfile("./data/segmantation_mask/seg_hyouzi_inf/"+GT) == True:
                seg_img3 = np.array(Image.open("./data/segmantation_mask/seg_hyouzi_inf/"+GT).convert('L'))
                c = c + 1
            else :
                seg_img3 = np.array(np.full((50,50), 255, dtype=np.uint8))
                
                
            if os.path.isfile("./data/segmantation_mask/gen_hall_inf/"+GEN) == True:
                seg_img1 = np.array(Image.open("./data/segmantation_mask/gen_hall_inf/"+GEN).convert('L'))
                d = d + 1
            if os.path.isfile("./data/segmantation_mask/gen_shadow_inf/"+GEN) == True:
                seg_img2 = np.array(Image.open("./data/segmantation_mask/gen_shadow_inf/"+GEN).convert('L'))
                e = e + 1

            if os.path.isfile("./data/segmantation_mask/gen_hyouzi_inf/"+GEN) == True:
                seg_img3 = np.array(Image.open("./data/segmantation_mask/gen_hyouzi_inf/"+GEN).convert('L'))
                f = f + 1
#            print(str(a),str(b),str(c))
#            print(str(d),str(e),str(f)+"\n\n")
    
            grayImgData = np.asarray(np.float32(img)/255.0)
            seg1  = np.asarray(np.float32(seg_img1)/255.0)
            seg2  = np.asarray(np.float32(seg_img2)/255.0)
            seg3  = np.asarray(np.float32(seg_img3)/255.0)
            
            
            imgData = np.asarray([grayImgData,seg1,seg2,seg3])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))

#            print(seg2[0],seg2[1],seg2[2])
            
        threshold = len(imageData) -100 
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        np.save('./np_data/image_95-5_'+num+mode+'.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'.npy', labelData)

        

        print(len(imageData))
        print(threshold)
        return train,test 
            
    if channels == 3:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
            img = Image.open(pathAndLabel[0]) 
            #3チャンネルの画像をr,g,bそれぞれの画像に分ける
            r,g,b = img.split()
            rImgData = np.asarray(np.float32(r)/255.0)
            gImgData = np.asarray(np.float32(g)/255.0)
            bImgData = np.asarray(np.float32(b)/255.0)
            imgData = np.asarray([rImgData, gImgData, bImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))

        np.save('./np_data/image_95-5_'+num+mode+'.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'.npy', labelData)
        threshold = len(imageData) -100 
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])
        print("train:"+str(len(imageData)))
        print("value:"+str(len(imageData)-threshold))
        return train,test 
           
    
    if channels == 21:
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%5000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))

    
            img41 = np.array(Image.open(pathAndLabel[0]).convert('L') )    


            if int(pathAndLabel[1]) == 0 :

                path31 =pathAndLabel[0].replace('median41', 'median21')
#                   path51 =pathAndLabel[0].replace('median41', 'median41')
            if int(pathAndLabel[1]) == 1 :

                path31 =pathAndLabel[0].replace('median41', 'median21')
#                   path51 =pathAndLabel[0].replace('median41', 'median41')

            img21 =  np.array(Image.open(path31).convert('L') )
            
            
#            epsilon = 0.5
#            if epsilon > random():

                
            grayImgData41 = np.asarray(np.float32(img41)/255.0)
            grayImgData21 = np.asarray(np.float32(img21)/255.0)
            imgData = np.asarray([grayImgData41,grayImgData21])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))  
#            else:
#                select_rate = select_rate + 1
#        print("Through"+str(select_rate))               
        np.save('./np_data/image_95-5_'+num+mode+'.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'.npy', labelData)
        threshold = len(imageData) -100 
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])

        print(len(imageData))
        print(threshold)        
        return train,test    
        
        
    if channels == 4:
    
        imageData = []
        labelData = []
        for pathAndLabel in allData:
            load_num = load_num + 1
            load_rate = load_rate+1.0
            if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
            img = Image.open(pathAndLabel[0])
            path = pathAndLabel[0][len(label0):]
            
            if int(pathAndLabel[1]) == 0 :
                path =pathAndLabel[0].replace('rgb', 'median41')
            if int(pathAndLabel[1]) == 1 :
                path =pathAndLabel[0].replace('rgb', 'median41')
    #            print(path)
            img2 = np.array(Image.open(path).convert('L') )
    
            #3チャンネルの画像をr,g,bそれぞれの画像に分ける
            r,g,b = img.split()
            rImgData = np.asarray(np.float32(r)/255.0)
            gImgData = np.asarray(np.float32(g)/255.0)
            bImgData = np.asarray(np.float32(b)/255.0)       
            grayImgData = np.asarray(np.float32(img2)/255.0)
            imgData = np.asarray([rImgData, gImgData, bImgData,grayImgData])
            imageData.append(imgData)
            labelData.append(np.int32(pathAndLabel[1]))        
#        threshold = np.int32(len(imageData)/8*7)
        np.save('./np_data/image_95-5_'+num+mode+'+median41.npy', imageData)
        np.save('./np_data/label_95-5_'+num+mode+'+median41.npy', labelData)
        threshold = len(imageData) -100 
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test  = tuple_dataset.TupleDataset(imageData[threshold:],  labelData[threshold:])

        print(len(imageData))
        print(threshold)      
        return train,test
    if channels == 5: #チャンネル数が1つの場合
            imageData = []
            labelData = []
            for pathAndLabel in allData:
                load_num = load_num + 1
                load_rate = load_rate+1.0
                if load_num%1000 ==0 : print ("load_rate={:.4}%\n\n".format(load_rate/number*100))
    #            print(pathAndLabel[0])
    
#                if int(pathAndLabel[1]) == 1 :
                img = np.array(Image.open(pathAndLabel[0]).convert('L') )
                grayImgData = np.asarray(np.float32(img)/255.0)
                imgData = np.asarray([grayImgData])
                imageData.append(imgData)
                labelData.append(np.int32(pathAndLabel[1]))
    
                
            print(len(imageData))
            train = np.asarray(imageData)
            return train     
