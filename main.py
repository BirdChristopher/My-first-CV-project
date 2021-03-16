# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//10
"""
from config import opt
import fire
import os
import torch as t
import models
from torch.utils.data import DataLoader
from data.dataset import dogcat
from utils.visualize import Visualizer
from torchnet import meter
from tqdm import tqdm

@t.no_grad()#被这个语句修饰的函数将不会被track的梯度，不会进行反向传播，grad_fn也不会被记录，目的是为了提高效率吧
def test(**kwargs):
    opt._parse(kwargs)

    #configure model
    model = getattr(models,opt.model)().eval()#这是一个工程方面的细节，可以注意到models是本文件目录的一个有__init__.py的包，该文件中
    if opt.load_model_path:#import了models包内的网络对应的类，那么在getattr语句就是找到这个类，
        model.load(opt.load_model_path)#找到的类后直接加()就是初始化一个该类的实例对象，再加一个.eval()就是改成在测试集上的验证模式
    #加载预训练模型的接口在BasicModule模块里建立好了
    model.to(opt.device)



    #data
    train_data = dogcat(opt.load_train_path,test = True)
    test_dataloader = DataLoader(train_data,batch_size = opt.batch_size,shuffle = False,num_workers = opt.num_workers)
    #dataset的作用是将所有的数据初始化加载，而dataloader则可以将dataset的初始化数据拼接成patch，dataloader是一个可迭代对象。
    #Dataloader有一个参数collate_fn，参数的输入是一个函数，表示的是拼接的方式，事实上我们可以自定义一款函数，来剔除一些加载不了的图片
    #自定义方法链接：https://blog.csdn.net/weixin_42464187/article/details/104795574

    results = []
    for ii,(data,path) in tqdm(enumerate(test_dataloader)):# tqdm(list)则可以打印list迭代过程中的进度,这里ii是下标，(data.path)才是真正的数据
        input = data.to(opt.device)#注意，从Dataloader里面弄出来的数据都是以batch形式组织的
        score = model(input)
        probability = t.nn.functional.softmax(score,dim = 1).detach().tolist()#不知道在干什么
        #注意softmax的参数，dim = 1是沿着第1个维度计算，当然一般来说只要网络的classifier接口设置好了，这里的dim参数一律填1
        '''     
        softmax函数将会把输出的数据转化为相加结果为1的概率形式，并使较大的得分的优势更加明显
        '''
        batch_result = [(path_.item(),probability_) for (path_,probability_) in zip(path,probability)]#这里有一个细节，
        #就是zip其实是可以作用于tensor或者ndarray的，然后将path这个tensor拆成一个一个shape为(1)的tensor，还需要用一个item()才能
        #真正提取其中的数据

        results += batch_result

    write_csv(results,opt.result_file)
    return results

def write_csv(results,filename):
    import csv
    with open(filename,"w") as f:
        writer = csv.writer(f)
        writer.writerow(['data','label'])
        writer.writerows(results)

def train(**kwargs):
    opt._parse(kwargs)

    visual = Visualizer(env=opt.env, port=opt.vis_port, use_incoming_socket=False)
    # step1 : configure model
    model = getattr(models,opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)#加载的接口已经封装好了，我们不用自己写加载的完整语句
    model.to(opt.device)

    #data
    model_for_val = getattr(models,opt.model)().eval()
    train_data = dogcat(opt.load_train_path,train = True)
    val_data = dogcat(opt.load_train_path,train = False)
    train_dataloader = DataLoader(train_data,batch_size= opt.batch_size,shuffle = True,num_workers= opt.num_workers,
                                      drop_last= False) # 训练集需要打乱，否则学习的结果可能受排列的规律影响
    val_dataloader = DataLoader(val_data,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers,
                                    drop_last=True)

    #criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = t.optim.Adam(model.parameters(),#参数的第一条是指定对哪些参数优化，含有一个可选参数是制定规则将不用优化的参数去掉
                             lr = lr,           #注意不要把整个model传进去，我们只需要model的参数即可
                             weight_decay = opt.weight_decay)

    #meter是一个从torchnet库导入的模块，提供统计平均值和统计混淆矩阵服务
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e100#设定最开始的loss，当然我们需要设置的尽量高些，方便边界条件成立。

    for epoch in range(opt.max_epoch):
        loss_meter.reset()
        confusion_matrix.reset()#重新设置这两个统计矩阵
        for ii,(data,path) in tqdm(enumerate(train_dataloader)):
            #训练模型参数
            input = data.to(opt.device)
            target = path.to(opt.device)#注意要把答案也放到device上去

            #设计好所有的接口，接下来的五个句子才是最核心的逻辑
            optimizer.zero_grad()#一次epoch之后，backward会导致models的参数梯度残留，必须要用zero-grad去除
            score = model(input)
            loss = criterion(score,target)#这个函数的实现比较复杂，展示只需要知道criterion的参数怎么填就好
            loss.backward()#model中backward自动生成，但是要手动调用
            optimizer.step()
            #meters update
            loss_meter.add(loss.item())#这个东西每次接受一个数据，调用时返回所有曾经接受的数据
                                        #的均值和标准差
            confusion_matrix.add(score.detach(),target.detach())#用score和target作混淆矩阵
                                                        #注意如果不是在前向传播的时候用tensor，最好都加上detach()
            if ii%opt.print_freq == opt.print_freq-1:
                visual.plot('loss',loss_meter.value()[0])#plot的接口已经设计好了
                #进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save()
        val_cm,val_accuracy = val(model,val_dataloader)

        visual.plot('val_accuracy',val_accuracy)#绘制准确率曲线
        visual.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch,lr=lr,loss=str(loss.value()[0]),val_cm=str(val_cm.value()),train_cm=confusion_matrix.value()
        ))

        #update learning rate
        if loss_meter.value()[0] > previous_loss:
            lr = lr*opt.lr_decay
            # 第二种降低学习率的方法，不会有momentum等信息的丢失？？？？？？
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]#为什么要加序号呢？因为loss_meter不仅返回均值，还返回标准差


@t.no_grad()
def val(model,dataloader):
    '''
    验证集验证逻辑
    '''
    model.eval()#更改模式
    confusion_matrix = meter.ConfusionMeter(2)
    for ii,(val_input,path) in tqdm(enumerate(dataloader)):
       val_input = val_input.to(opt.device)
       score = model(val_input)
       confusion_matrix.add(score.detach().squeeze(),path.type(t.LongTensor).detach())
            #经过试验，这样的格式转换似乎没有必要

    model.train()#改回来
    cm_value = confusion_matrix.value()#将混淆矩阵的结果导出到一个变量中，这样方便我们取出矩阵中的值
    accuracy = 100.*(cm_value[0][0] + cm_value[1][1])/(cm_value.sum())#两个接口：一个是.value（）返回混淆矩阵，一个是.sum（）返回矩阵所有值之和
    return confusion_matrix,accuracy

def help():#怎么写帮助文件？？学学！
    '''
    打印帮助信息
    '''
    print("""
    usage: python file.py <function> [--args == value]
    <function> : = train | test | help
    example:
        python {0} train --env ='env0701' --lr=0.01
        python {0} test --dataset='path/to/dataset/root/'
        python {0} help
    available args:""".format(__file__))#__file__的返回值是该文件的存储位置，那么单独的一个__file__返回什么意思呢？？

    from inspect import getsource
    source = (getsource((opt.__class__)))# getsource()的返回值是以字符串形式返回源代码，opt是setDefaultConfig的实例对象
    print(source)

if __name__ == '__main__':
    fire.Fire()