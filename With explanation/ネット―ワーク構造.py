# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 09:14:46 2018

@author: g089v
"""

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers
from chainer import training
from chainer.datasets import cifar
from chainer import iterators
from chainer.training import extensions
from chainer import serializers
import numpy as np
import matplotlib.pyplot as plt

#ネットワークの構造の定義
class CNN(Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(CNN, self).__init__(
            conv1  = L.Convolution2D(None,16, ksize=3, pad=1,initialW=initializer),
            conv3  = L.Convolution2D(None,32, ksize=3, pad=1,initialW=initializer),
            conv5  = L.Convolution2D(None,64, ksize=3, pad=1,initialW=initializer),
            conv7  = L.Convolution2D(None,128, ksize=3, pad=1,initialW=initializer),
            conv9  = L.Convolution2D(None,256, ksize=3, pad=1,initialW=initializer),
            conv11 = L.Convolution2D(None,512, ksize=3, pad=1,initialW=initializer),
            conv13 = L.Convolution2D(None,1024, ksize=3, pad=1,initialW=initializer),
            conv14 = L.Convolution2D(None,1024, ksize=3, pad=1,initialW=initializer),
            conv15 = L.Convolution2D(None,1024, ksize=3, pad=1,initialW=initializer),
            l16 = L.Linear(1024,256),
            l17 = L.Linear(None,128),
            l19 = L.Linear(None,2),
        )

    def __call__(self,x):

        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.relu(self.conv5(h))
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.relu(self.conv7(h))
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.relu(self.conv9(h))
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)
        h = F.relu(self.conv11(h))
        h = F.max_pooling_2d(h,ksize=2,stride=2,pad=0)


        h = F.relu(self.conv13(h))
        h = F.relu(self.conv14(h))
        h = F.relu(self.conv15(h))
        h = F.relu(self.l16(h))
        h = F.relu(self.l17(h))
        # skip dropout
        h = self.l19(h)

        return h



        
#学習に関する関数（chainerのtrainerを参照）        
def train_CNN(network_object, batchsize=128, gpu_id=-1, max_epoch=20, train_dataset=None, test_dataset=None, postfix='', base_lr=0.01, lr_decay=None,number = 11):
    number = str(number)
    # 1. Dataset
    if train_dataset is None and test_dataset is None:
        train, test = cifar.get_cifar10()
    else:
        train, test = train_dataset, test_dataset
        
    if gpu_id >= 0:
        network_object.to_gpu(gpu_id)
    # 2. Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)

    # 3. Model
    net = L.Classifier(network_object)

    # 4. Optimizer
    optimizer = optimizers.MomentumSGD()
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # 5. Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # 6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_crack_{}result'.format(network_object.__class__.__name__, postfix))
    
    # 7. Trainer extensions
    trainer.extend(extensions.LogReport(trigger=(1, 'epoch'), log_name="log_"+number))
    trainer.extend(extensions.snapshot(filename=number+'snapshot_epoch-{.updater.epoch}'),trigger=(5, 'epoch'))
#    trainer.extend(extensions.snapshot(), trigger=(10, 'epoch'))
    trainer.extend(extensions.ParameterStatistics(net.predictor.conv1, {'std': np.std}))
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(test_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss'+number+'.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy'+number+'.png'))
    trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='std'+number+'.png'))    
    if lr_decay is not None:
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_decay)
    trainer.run()
    del trainer

    return net                
   

    
        
        
#    
model = CNN()

