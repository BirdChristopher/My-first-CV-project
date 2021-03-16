# -*- coding:utf-8 -*-
"""
author:Bird Christopher
date:2021//03//07
"""
import visdom
import time
import numpy as np
# 可别忘了用visdom之前要先开启服务器！python -m visdom.server
# env：不同环境的可视化结果相互隔离，互不影响，在使用时如果不指定env，默认使用main。不同用户、不同程序一般使用不同的env。就是说开一个本地服务器，
# 可以同时显示多个程序的可视化结果
# https://github.com/fossasia/visdom是这个库的Github仓库
class Visualizer(object):
    '''
    这个类中封装了visdom的基本操作，当然仍然可以通过self.vis.function来调用原生接口
    '''

    def __init__(self,env = 'default',**kwargs): # 一个细节：在构造方法中使用了**kwargs，这样我们在下面使用这些参数的时候就不需要一个一个
                                                 # 写出来
        self.vis = visdom.Visdom(env = env,use_incoming_socket = False,**kwargs)#构建一个客户端对象，这个对象有很多方法可以用
        # env就是说指定的environment的名称，这里注册为env，其他的可能注册为env2啥的
        # use_incoming_socket:启用套接字以从Web客户端接收事件，允许用户注册回调(默认值:"True")
        # 一般这个参数就设置为False就好了
        self.index = {}# 先初始化index属性为一个字典！！！注意不是集合，log_text为一个字符串
        self.log_text = ""# 为什么不在实际使用的时候在创建这两个变量呢？

    def reinit(self,env = 'default',**kwargs):
        '''
        作用：修改visdom的配置
        '''
        self.vis = visdom.Visdom(env = env,**kwargs)
        return self

    def plot_many(self,d):# 面向对象编程的技巧：可以设置一个方法用于重复调用另一个方法
        '''
        一次plot多个
        :param d: dict(name,value) i.e.('loss',0.11)
        :return:
        '''
        for k,v in d.items():# 对于字典类型d，d.items()以列表形式返回d中的所有键值对
            self.plot(k,v)
    def img_many(self,d):
        for k,v in d.items():
            self.img(k,v)

    def plot(self,name,y,**kwargs):#name用于指定在哪一个曲线上操作，y表示新传入的值
        x = self.index.get(name,0)# Python字典的操作函数.get（key,default）,根据键key返回对应的值，若找不到则返回default
        self.vis.line(Y=np.array([y]),
                      X=np.array([x]),# visdom能接受的传入数据只能是tensor或者是ndarray，也许这个操作的意义仅在于将类型转化为ndarray？
                      win = name,# 用于指定显示此图像的pane的名称
                      opts = dict(title = name),# 接受一个字典，根据字典中的键值对设置pane，比如设置title，width，xlabel等等
                      update = None if x == 0 else 'append',#设置update=append，可以使新绘制的点不覆盖原图，而是追加
                    # 而如果设置为new的话，就是在一个pane中再画一条新曲线
                      # update = None看起来和new是一个意思
                      #如果想要动态的生成曲线以展示过程，就一次画一个点，设置update= append，注意转换格式为tensor或者ndarray
                      **kwargs
                      )
        self.index[name] = x+1 # 太凝练了。。若不存在键叫做name，则创建一个，并且x之前取出来就是0
                                #如果存在就将对应的键值加一，表示又在这个曲线上加了一个新点，下一个点的横坐标为该点的横坐标+1
                                #这里用字典来管理不同曲线绘制进度，使得一个接口可以绘制不同的曲线，只要传入的name不同。。。太强了！！！

    def img(self,name,img_,**kwargs):
        self.vis.image(img_.cpu().numpy(),#将img_强行转换为ndarray
                       win = name,#注意img_传入时就应该是一个tensor，如果是要显示彩图的话
                       opts = dict(title = name),#转换为字典，跟直接用{}差不多
                       **kwargs)

    def log(self,info,win='log_text'):#注意，log_text遵从html语法标准，换行得用<br>而不是\n
        self.log_text += ('[{time}]{info}<br>'.format(
            time = time.strftime("%m%d_%H%M%S"),
            info = info#这里的{time}，{info}用到了python字符串格式化的语法
        ))#这里的细节注意！我们在构造方法中已经定义了log_text是一个空串，这里就可以很方便的用+号来管理
        #输出，而不需要额外判定是否log_text已被初始化，牛逼！
        self.vis.text(self.log_text,win)#update默认是覆盖原内容

    def __getattr__(self, name):
        return getattr(self.vis,name)#精妙的小技巧，通过定义__getattr__使得查找参数时还可以返回visdom对象的参数，没有这个的话只能返回这个类的属性
                                    #蛮好的一个习惯的，当一个类的定义中创建了另一个类的对象时，自定义一个__getattr__是很有用的
    #__getattr__方法作用于属性查找的最后一步，如果没有找到对应属性并且类中没有定义__getattr__，那么就会报错
    '''
    vis = visdom.Visdom(env=u'test1')，用于构建一个客户端，客户端除指定env之外，还可以指定host、port等参数。
    vis作为一个客户端对象，可以使用常见的画图函数，包括：
    
    line：类似Matlab中的plot操作，用于记录某些标量的变化，如损失、准确率等
    image：可视化图片，可以是输入的图片，也可以是GAN生成的图片，还可以是卷积核的信息
    text：用于记录日志等文字信息，支持html格式
    histgram：可视化分布，主要是查看数据、参数的分布
    scatter：绘制散点图
    bar：绘制柱状图
    pie：绘制饼状图
    '''