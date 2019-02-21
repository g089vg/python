from simple_convnet3 import CNN,train_CNN
from F1measure import F1_measure
from load_dataset import Load_Dataset
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    
    train,test = Load_Dataset(label0="../holdout/3/inflation_0_rgb/",label1="../holdout/3/inflation_1_rgb/",channels = 1)

    size = 50
    x,t = train[1]
#    plt.imshow(x.reshape(size, size), cmap='gray')
    plt.imshow(x.reshape(size, size))
    plt.show()
    print(t)

#    plt.show()
    
    network = CNN()
    gpu_id = -1
    
    
    train_CNN(network, batchsize=50, gpu_id = -1, max_epoch=100, train_dataset = train, test_dataset = test)
    
    
    
    
