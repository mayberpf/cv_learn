# YOLOv5 module
最近也是一直在看YOLOv5的模型，这里就记录一下在看代码学到的一些新的东西。主要是也方便后面我改代码。
本代码不是YOLOv5的官方代码，而是参考b站的一个博主的。
参考链接：【Pytorch 搭建自己的YoloV5目标检测平台（Bubbliiiing 源码详解 训练 预测）-YoloV5整体结构介绍】 https://www.bilibili.com/video/BV1FZ4y1m777?share_source=copy_web&vd_source=7af49fc7f4527a070bbd7df512e8c736
### 网咯的冻结
参考：http://t.csdn.cn/4gbPH
目前在这个代码中，冻结只针对网络的backbone进行，训练开始之前会设置多个epoch，分别是：Init_Epoch   Freeze_Epoch      UnFreeze_Epoch、Freeze_Train。也就是起初的训练会将backbone冻结，也就是backbone不会学习权重，当epoch大于Freeze_Epoch小于UnFreeze_Epoch，就不再进行网络冻结了，也就是网络模型都可以学习权重。
### SiLU
为了进一步了解该激活函数，查了下资料。参考：http://t.csdn.cn/4rt4q
这篇文章讲了25种激活函数。本篇文章只分析SiLU
### 正则化
参考：http://t.csdn.cn/dXP2m
之前对于欠拟合过拟合进行过总结，也提到过针对过拟合的其中一个解决方法就是正则化。
### EMA
参考：http://t.csdn.cn/jy4uP
指数移动平均
### 训练过程中的记录



### YOLOloss



### 优化器 学习率


### weight_init

