# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//03
"""
import os
from torchvision import transforms as t
from torch.utils import data
import numpy as np
import PIL

class dogcat(data.datasets):#有一个问题是，代码中并未涉及如果transforms不是None时的处理方式
    def __init__(self,root,transforms = None,train = True,test = False):#留下了很多接口！
        self.test = test
        self.train = train
        pth_imgs = [os.path.join(root,img_name) for img_name in os.listdir(root)]#非常精巧的一句表达！

        if self.test:
            imgs = sorted(pth_imgs,lambda x:int(x.split('.')[-2]).split('/')[-1])#这个排序不进行会不会出问题呢？
        else:
            imgs = sorted(pth_imgs,lambda x:eval(x.split(".")[-2]))#用eval也是一样的吧？

        imgs_len = len(imgs)

        if self.test:
            self.imgs = imgs
        elif self.train:
            self.imgs = imgs[0:int(0.7*imgs_len)]#截取imgs数组的前70%用作训练集
        else:
            self.imgs = imgs[int(0.7*imgs_len):]#截取imgs数组的后30%用作验证集

        if transforms == None:#如果初始化时没有定义transforms，就会按照接下来规定的默认方式进行初始化处理
            normalize = t.Normalize(mean = [0.485,0.456,0.406],std = [0.229,0.224,0.225])#注意，normalize的功能是将每个值减去mean，再
        #再除std，目的是将各个元素归一化到[-1,1]上。至于为什么选取这六个值，因为这是ImageNet中全体样本的均值和方差，这么做就不用了自己算自己的样本了

        if train == True:
            self.transform = t.Compose(#搞这么多transforms的目的，一方面是为了使输入图片的格式统一，另一方面是为了提升泛化能力
                [t.Resize(256),
                 t.RandomResizedCrop(224),
                 t.RandomHorizontalFlip(),
                 t.ToTensor,
                 normalize
                ])
        else:
            self.transforms = t.Compose(
                [t.Resize(224),#注意，这个变换是按比例把最短边设置为224个像素！
                 t.CenterCrop(224),
                 t.ToTensor(),
                 normalize
                ]
            )
        #一些注意事项：self.transforms的输入应该是一个image数据（就是用PIL.Image(path)打开的数据），
        #不需要提前将image转为tensor，其会在transforms的过程中被转换为tensor

        def __getitem__(self,index):
            img_pth = self.imgs[index]
            data = PIL.Image.open(img_pth)
            data = self.transforms(data)
            if self.test:#暂时没有搞明白为什么测试集的数据需要在名字中把种类信息去掉？非常bewildering
                label = img_pth.split(".")[-2].split("/")[-1]
            else:
                label = 1 if "dog" in img_pth.split(".")[-1] else 0
                #又是一句及其精简的表达，注意这种将if else 简化在一句话之内的表达
            return data,label

        def __len__(self):
            return  len(self.imgs)



'''
tips:可以注意到，我们在定义dataset的子类时，会在__init__构建方法中初始化所有图片的地址，这个操作不是很耗时，
但是根据地址加载这些图片是非常耗时的，我们一般会将真正加载图片的操作放到__getitem__方法中，这样可以提高加载速度



我们通过dataset定义的类虽说是一个迭代器，但是一次只能根据下标返回一个元素，而dataloader类则可以将元素拼接成batch，并且利用并行加速，所以
我们还需要定义一个dataloader，在这个工程中dataloader被我们安置在main函数中定义。 
'''