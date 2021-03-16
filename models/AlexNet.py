# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//05
"""
import torch as t
from torch import nn
from torch.utils import data
import os
import numpy as np
import visdom
import torchvision
from .BasicModule import BasicModule
class AlexNet(BasicModule):
    def __init__(self,num_classes = 2):#注意训练这个网络用于图片分类，还需要分类种数这一参数
        super(AlexNet,self).__init__()
        self.model_name = "AlexNet"
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=11,stride = 4,padding = 0),#padding参数的意义：输入的每一条边补充0的层数
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3,stride = 2),
            nn.Conv2d(64,192,kernel_size=5,padding = 2),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size= 3,stride = 2),# 为什么有些卷积层后面跟了池化层，有些没有？
            nn.Conv2d(192,384,kernel_size = 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384,256,kernel_size=3,padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(256,256,kernel_size= 3,padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3,stride = 2)
        )
        #像卷积->非线性->池化是一种最优的策略吗？

        self.classifier = nn.Sequential(#为什么dropout层不放到feature层去？
            nn.Dropout(),#为什么Dropout在classifier中出现
            nn.Linear(256*6*6,4096),#这个参数是什么意思？
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096,4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096,num_classes)
        )

    def forward(self,input):
        input = self.features(input)
        input = input.view(input.size(0),256*6*6)#将256*6*6改成-1会有问题吗？？？这个256*6*6怎么来的？
        # 这个语句一般就用在forward过程中的分类器之前，目的是将多维的数据展成一维
        # 已经经过Dataloader初始化的数据的组织形式是（batchsize，channel，w，h）,所以input.size（0）的意思是batchsize
        input = self.classifier(input)
        return input