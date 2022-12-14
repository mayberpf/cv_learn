# 19
## 随机梯度下降(SGD)、小批量梯度下降(mBGD)、批量梯度下降(BGD)各自的优缺点
首先关于这一部分详细解释可以参考：http://t.csdn.cn/ixW4I
看了文章之后，我的理解，就是随机梯度下降法是针对一个样本进行计算，虽然速度快，但是准确度却不高。况且在山谷或者鞍点会一致震荡，很难出来。那么批量梯度下降法就相反，他每次的梯度计算都会考虑全部的数据集，这就导致计算时间和内存很大。
因此，在随机梯度下降中，相当于batchsize==1，学习率要设置的比较小，但是由于这样会对gpu有浪费，因为batchsize小很难占满整个gpu。
批量梯度下降除了数据量大，计算耗时，又因为神经网络是非凸的，网络最终可能收敛到初始点附近的局部最优点，不太容易达到较好的收敛性能。
那么mBGD实际上就是对上面两个的优化。


## 为什么batch size 很重要
一次训练选取的样本数，它的大小影响模型的优化程度和速度

## batch size 越大对模型来说有哪些好处
大的batchsize可以提高内存使用率，加快模型的训练，减少震荡，梯度下降的方向更准确。

## 大batchsize好，还是小batchsize好，又或者说太大太小都不合适，而取中间值更好
没有batchsize，梯度准确，只适用于小样本数据
batchsize==1，梯度变来变去，非常不准确，网络很难收敛
batchsize增大，梯度变准确
但是当batchsize增大到一定程度，梯度已经很准确了，再增大就没有意义了。

## batchsize和学习率的关系是什么
batchsize越大，学习率一般也要变大

## 对于多GPU，实际的batchsize是什么

官方的解释是：batch的数量会均分到每块GPU上进行处理，因此要保证每一个整数的关系
也就是两张卡batchsize==64，那么每张卡就是32。