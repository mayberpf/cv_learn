# 16
### 1、准确率（accuracy）的计算公式是什么？错误率（Error）的计算公式是什么？
### 2、用pytorch简要实现一下准确率的算法
### 3、召回率（Recall）的计算公式是什么？准确率（precision）的计算公式是什么？
### 4、如何理解P-R曲线？如何计算精度，平均精度（AP，mAP）？如何计算F1分数？
### 5、如何理解ROC曲线？AUC曲线？

#### 1
首先清楚以下几个概念：
TP：实际为正，预测为正
FP：实际为负，预测为正
FN：实际为正，预测为负
TN：实际为负，预测为负
助记：F== Fales、T== True、P== positive、N==negative
那么准确率accuracy ==(TP+TN)/(TP+TN+FP+FN)
错误率error ==(TN+FN)/(TP+TN+FP+FN)

#### 2
我感觉这个模块需要分不同的情况，目标检测，图像分割要分开的吧

#### 3
recall召回率：(TP)/(TP+FN)====我需要检测到的物体中实际检测到的比例
precision精确率：(TP)/(TP+FP)====就是我实际检测到的物体中，正确的比例

#### 4
PR曲线就是精确率和召回率的的曲线，横坐标为recall，纵坐标为precision
AP是平均精确度，是对PR曲线上的precision求平均值，所以AP是针对一种类别，而mAP是针对所有类别的平均值
F1-score又叫做平衡F分数，计算公式为：2 * precision * recall/(precision+recall)

#### 5
ROC曲线是根据不同的阈值下的recall和precision，得到的曲线，是用来衡量分类器的分类能力的。AUC是ROC曲线下方的面积。ROC的含义为概率曲线，AUC的含义为正负类可正确分类的程度。一般来说，AUC越靠近1，效果越好，而ROC越靠近左上角，效果越好。
