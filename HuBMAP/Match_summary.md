#HuBMAP
官网：https://www.kaggle.com/competitions/hubmap-organ-segmentation
并没有分patch----因为分辨率只有3000*3000
但是一开始我们做了切patch
数据 编码的方式---numpy2rle
第二名用了external data 但是应该美起到什么作用

transformer，dataset，dataloader
albumentations 库
手写的数据增强---
色彩迁移：根据目标图片调整图片的颜色特征   hsv

这里没有做

lib：segmentation_model_pytorch as smp
预训练---完形填空

timm库
第二名的代码要看？？？60个模型？

第四名
对于难识别的类
没开元？

使用框架

自己用了非常重的encoder

loss---nn.bce
diceloss

第二名
px_size
目标检测
多任务loss

metric
dice，iou

train_one_epoch

test_one_epoch
trick
    k-fold
    model-ensemble
    tta
    pseudo label
    stain normalization

后处理算法
    CRF
    分水岭


简历书写
数据集--->transformer--->
数据集是什么+增强+模型+loss+推理+指标
+个人博客地址
github repo

技能特长
每个项目都要有描述自己做了什么
数据集有多少张，最终的metric多少

cutmix？？
目标检测很有用？
写一些你懂的，有把握的，不要写有歧义的
cutout？？？

