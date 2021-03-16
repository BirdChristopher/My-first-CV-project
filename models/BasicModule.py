# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//06
"""
#本类的目的是提供快速加载和保存模型的接口。
import torch as t
from torch.utils import data
import numpy as np
import time

class BasicModule(t.nn.Module):
    def __init__(self):
        super(BasicModule,self).__init__()
        self.module_name = str(type(self))

    def load(self,path):
        self.load_state_dict(t.load(path))

    def save(self,name = None):
        '''
        默认按照 名字+修改时间的方式保存
        '''
        if name == None:
            prefix = 'checkpoints/'+self.module_name
            name = prefix + time.strftime(prefix,"%y.%m.%d.%H:%M:%S.pth")#若这里用+号，则是无空格地连接字符串

        t.save(self.state_dict(),name)
        return name

# 这里建议保存对应的state_dict，而不是直接保存整个Module/Optimizer对象。
# Optimizer对象保存的主要是参数，以及动量信息，通过加载之前的动量信息，能够有效地减少模型震荡