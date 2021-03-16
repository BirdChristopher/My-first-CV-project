# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//07
"""
from torchvision.models import squeezenet1_1
from .BasicModule import BasicModule
from torch import nn
from torch.optim import Adam

class SqueezeNet(BasicModule):
    def __init__(self,num_classes = 2):
        super(SqueezeNet, self).__init__()
        self.model_name = 'SqueezeNet'
        self.model = squeezenet1_1(pretrained=True)#这样的已训练好的模型都在torchvision.models里面
                                #直接调取后还可以通过pretrained参数来决定是用训练好的网络，
                                # 还是只用其一个模型。一般来说分类器部分是需要自己手写的，因为分类器部分
                                # 的参数值随问题的情况而改变
        self.model.num_classes  = num_classes#这种给类的属性定义次一级属性的做法常用吗？
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512,num_classes,1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13,stride = 1)
        )

    def forward(self,x):
        return self.model(x)

    def get_optimizer(self,lr,weight_decay):#调用这个方法，返回一个optimizer
        return Adam(self.model.classifier.parameter(),lr,weight_decay = weight_decay)

    #weight decay是用于防止过拟合的，系数越大，干扰越大
    # Adam接受可以优化的参数（必须都是variable对象）作为第一个参数，之后可以设置学习率和权重衰减值
    # 事实上，我们还可以为每一层网络分别设置optimizer的参数，但是此时的传入必须是一个dict，且params作为
    # 键值。具体形式如下：
    #   optimizer =optim.SGD([
    #            {'params': net.features.parameters()}, # 学习率为1e-5
    #            {'params': net.classifier.parameters(), 'lr': 1e-2}
    #       ], lr=1e-5)
    # 在调用了backward之后用optimizer.step()来进行一次优化
