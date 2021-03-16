# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//09
"""
import warnings
import torch as t
class DefaultConfig(object):#所有的Python类都继承自object基类
    model = "AlexNet"
    vis_port = 8097
    load_train_path = "./trainimg/"#这个结尾的/是必须的，这与我们前面的代码逻辑必须符合，即直接将root与文件名合并，并未额外添加/
    load_test_path = "./test/"
    load_model_path = None #加载预训练的模型的路径，若为none则不加载
    env = "default"#visdom默认名称
    num_workers = 4 #加载数据的worker数目？？？？？？
    use_gpu = True
    batch_size = 128 #一个batch中有多少个图片？
    epoch = 4 #一次训练训练4个batch
    result_file = "result.csv"
    debug_file = "./tmp/debug" # ????if os.path.exists(debug_file): enter ipdb

    print_freq = 4 #每训练n个batch输出一次结果
    max_epoch = 10 #设置的意义？
    lr = 0.1
    lr_decay = 0.95#学习率衰减率
    weight_decay = 1e-4 #正则化系数
    def _parse(self,kwargs):
        for k,v,in kwargs.items():
            if not hasattr(self,k):#一个细节，这个函数不仅能提取实例属性，还能提取类属性
                warnings.warn("Warning:opt has no attribute %s" %k) #Python的格式化输出，用format更好(看起来更规整)
            setattr(self,k,v)

        opt.device = t.device('cuda') if opt.use_gpu else t.device('cpu')
        # 这一句话是为了tensor.to(device)做准备，这个t.to()语句的参数必须是t.device()返回的对象

        print('user config:')#这是一套组合用法：实例对象的__class__就是指其所属于的类，.__dict__是将
        for k,v in self.__class__.__dict__.items():# 该类或者类对象的参数和名字提取成一个一个字典，.items()则是将字典转为元组列表
            if not k.startswith('_'):
                print(k,':',getattr(self,k)) # 为什么不能直接用v呢？？？？

opt = DefaultConfig()
#一个基本的事实是，即使类的定义是写在opt的生成语句之前的，但其实类对象的代码在
#opt = DefaultConfig()语句执行时才会完整地执行
