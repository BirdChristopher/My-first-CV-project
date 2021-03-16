# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//06
"""
from .BasicModule import BasicModule
from  torch import nn
from torch.nn import functional as f
import torch

class ResidualBlock(nn.Module): # 注意残差块的特点，
    def __init__(self,inchannel,outchannel,stride =1,shortcut = None):
        super(ResidualBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,3,stride = 1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace = True),
            nn.Conv2d(outchannel,outchannel,3,1,1,bias = False),
            nn.BatchNorm2d(outchannel) # 为什么这里BatchNorm之后没有非线性激活函数了？
        ) # 左边学习的是残差函数，基本结构就是weightlayer+relu+weightlayer，这个结构似乎是固定的

        self.right = shortcut#一般来说right部分是跨层直连，传递identity部分

    def forward(self, input):
        output = self.left(input)
        residual = input if self.right == None else self.right(input)
        output = output+residual
        return output

class ResNet34(BasicModule):#可以注意到，真正的模型都是继承BasicModule，而不是nn.module
    def __init__(self,num_classes = 2):
        super(ResNet34, self).__init__()
        self.model_name = "ResNet34"#定义其名字属性是个好习惯

        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias = False),#刚开始前向传播的数据通道数当然是3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size=3,stride = 2,padding = 1)
        )

        self.layer1 = self._make_layer(64,128,3)
        self.layer2 = self._make_layer(128,256,4,stride = 2)
        self.layer3 = self._make_layer(256,512,6,stride = 2)
        self.layer4 = self._make_layer(512,512,3,stride = 2)
        #有一个细节，可以在构造方法__init__中使用写在后面的类方法！！
        self.fc = nn.Linear(512,num_classes)#别忘了全连接分类层！

    def _make_layer(self,inchannel,outchannel,block_num,stride = 1):
        '''
        构建layer，用函数建layer，很舒服
        block_num参数的意义是：这一层的残差块的数量，第一个残差块会特殊一点，跨层直连部分
        不是传递identity
        '''
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias = False),
            nn.BatchNorm2d(outchannel)
        ) # 为什么identity部分变成了卷积的结果？是为了让outchannel数相等吗？
        layers=[]# 先创建列表，再一个一个加进去
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))

        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)#这个加‘*’的操作是吧列表拆成一个一个的分离的元素，因为nn.Sequential bu不是接受列表作为参数的
            # 注意这个Python的*list技巧：能够做到批量生成一个函数的arguments
            # 有一个细节是：为什么layers列表中的元素都是实例对象，并不是什么nn.Conv2d,nn.BatchNorm2d
            # 之类的单一目的的层。
            # 事实上，ResidualBlock是一个继承自nn.Module且实现了forward方法的类，nn.Conv2d之类的也是继承自
            # nn.Module,实现了forward方法（源码中有），所以二者是同一性质的module，而ResidualBlock仅仅是
            # 进行了一种自定义并高层抽象了而已，所以当然可以这样操作，
    def forward(self,input):
        input = self.pre(input),
        input = self.layer1(input),
        input = self.layer2(input),
        input = self.layer3(input),
        input = self.layer4(input),
        x = f.avg_pool2d(input,7)
        x = x.view(x.size(0),-1)
        return self.fc(x)

