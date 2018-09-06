# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 08:53:45 2018

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

class CNN(Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(CNN, self).__init__(
            conv1=L.Convolution2D(None,  16, 5, stride=2,initialW=initializer),
            bn1=L.BatchNormalization(16),
            conv2=L.Convolution2D(16, 32,  5, pad=2,initialW=initializer),
            bn2=L.BatchNormalization(32),
            conv3=L.Convolution2D(32, 48,  3, pad=1,initialW=initializer),
            bn3=L.BatchNormalization(48),
            conv4=L.Convolution2D(48, 64,  3, pad=1,initialW=initializer),
            fc5=L.Linear(64, 2),
            
        )
#        self.dr1 = 0.2
        self.dr2 = 0.5
    def __call__(self, x):
        if chainer.config.train:
        # 学習中の処理
            h = self.conv1(x)
            h = self.bn1(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            h = self.conv2(h)
            h = self.bn2(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            h = F.dropout(h, ratio=self.dr2 )
            h = self.conv3(h)
            h = self.bn3(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            h = F.dropout(h, ratio=self.dr2 )
            h = self.conv4(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            h = F.dropout(h, ratio=self.dr2 )
            h = self.fc5(h)
        else:
#            print('data predict')
            h = self.conv1(x)
            h = self.bn1(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)
            h = self.conv2(h)
            h = self.bn2(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)

            h = self.conv3(h)
            h = self.bn3(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)

            h = self.conv4(h)
            h = F.max_pooling_2d(F.relu(h), 3, stride=2)

            h = self.fc5(h)
        return h

    def get_inter_layer(self, x):
            conv_1 = self.conv1(x)
            conv_1 = self.bn1(conv_1)
            pool_1 = F.max_pooling_2d(F.relu(conv_1), 3, stride=2)
            conv_2 = self.conv2(pool_1)
            conv_2 = self.bn2(conv_2)
            pool_2 = F.max_pooling_2d(F.relu(conv_2), 3, stride=2)
    
            conv_3 = self.conv3(pool_2)
            conv_3 = self.bn3(conv_3)
            pool_3 = F.max_pooling_2d(F.relu(conv_3), 3, stride=2)
    
            conv_4 = self.conv4(pool_3)
            pool_4 = F.max_pooling_2d(F.relu(conv_4), 3, stride=2)
    
            h = self.fc5(pool_4)
            print(h)
            return conv_1


        
        
def train_CNN(network_object, batchsize=128, gpu_id=-1, max_epoch=20, train_dataset=None, test_dataset=None, postfix='', base_lr=0.01, lr_decay=None):

    # 1. Dataset
    if train_dataset is None and test_dataset is None:
        train, test = cifar.get_cifar10()
    else:
        train, test = train_dataset, test_dataset

    # 2. Iterator
    train_iter = iterators.MultiprocessIterator(train, batchsize)
    test_iter = iterators.MultiprocessIterator(test, batchsize, False, False)

    # 3. Model
    net = L.Classifier(network_object)

    # 4. Optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(net)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

    # 5. Updater
    updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

    # 6. Trainer
    trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='{}_crack_{}result'.format(network_object.__class__.__name__, postfix))
    
    # 7. Trainer extensions
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(test_iter, net, device=gpu_id), name='val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    
    if lr_decay is not None:
        trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger=lr_decay)
      

    print("start train")
    trainer.run()
    serializers.save_npz("CNN_model.npz", net) # npz形式で書き出し
    del trainer

    return net                
   

    
        
        
        
model = CNN()


