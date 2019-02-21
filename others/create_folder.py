# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 09:31:30 2018

@author: g089v
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 15:55:20 2018

@author: g089v
"""

# coding: utf-8
import os



if __name__ == '__main__':



    data_dir_path = u"./data/half_data"
    file_list = os.listdir(r'./data/half_data/')
            
    nnum = 0
    for file_name in file_list:
        root, ext = os.path.splitext(file_name)
        if ext == u'.bmp':
            nnum = nnum + 1
            print(file_name)
            abs_name1 = data_dir_path + '/' + file_name
            file_name = file_name[:-4]
            
            
#            new_dir_path = './data/Ground_Truth/'+file_name
#            os.mkdir(new_dir_path)
#            
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1'
#            os.mkdir(new_dir_path)
            
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1/rgb'
#            os.mkdir(new_dir_path)            
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1/gray'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1/median11'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1/median21'
#            os.mkdir(new_dir_path)             
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1/median31'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1/median41'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/1/median51'
#            os.mkdir(new_dir_path)             
            
            new_dir_path = './data/Ground_Truth/'+file_name+'/1/inf_rgb'
            os.mkdir(new_dir_path)            
            new_dir_path = './data/Ground_Truth/'+file_name+'/1/inf_gray'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/1/inf_median11'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/1/inf_median21'
            os.mkdir(new_dir_path)             
            new_dir_path = './data/Ground_Truth/'+file_name+'/1/inf_median31'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/1/inf_median41'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/1/inf_median51'
            os.mkdir(new_dir_path) 
            
            
            
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0'
#            os.mkdir(new_dir_path)
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0/rgb'
#            os.mkdir(new_dir_path)            
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0/gray'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0/median11'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0/median21'
#            os.mkdir(new_dir_path)             
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0/median31'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0/median41'
#            os.mkdir(new_dir_path) 
#            new_dir_path = './data/Ground_Truth/'+file_name+'/0/median51'
#            os.mkdir(new_dir_path)                
            
            new_dir_path = './data/Ground_Truth/'+file_name+'/0/inf_rgb'
            os.mkdir(new_dir_path)            
            new_dir_path = './data/Ground_Truth/'+file_name+'/0/inf_gray'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/0/inf_median11'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/0/inf_median21'
            os.mkdir(new_dir_path)             
            new_dir_path = './data/Ground_Truth/'+file_name+'/0/inf_median31'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/0/inf_median41'
            os.mkdir(new_dir_path) 
            new_dir_path = './data/Ground_Truth/'+file_name+'/0/inf_median51'
            os.mkdir(new_dir_path) 
            
            # os.mkdir(new_dir_path)
            # FileExistsError: [Errno 17] File exists: 'data/temp/new-dir/'
