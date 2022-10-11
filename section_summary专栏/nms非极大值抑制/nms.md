# NMs非极大值抑制
NMS也叫做非极大值抑制，就是在目标检测中，不论是one-stage还是two-stage，都会 产生许多预测框，他们大多都指向同一个目标，因此需要通过极大值抑制来筛选掉多余的框，来找到每个目标最优的预测框。
基本做法：先选择置信度最高的框作为基准，计算其余框与该框的IOU重合度，如果IOU重合度超过阈值，说明这两个框可能是预测的相同的物体，所以就去掉置信度低的，保留高的。重复这个过程直到所有的框的重合度都低于阈值。使用该方法可以大量减少冗余的预测框。

缺陷：
阈值不好确定，阈值低了容易出现漏检，高了容易误检
计算IOU的方式有缺陷，如果两个框没有相交，根据定义，iou=0，不能反应两者之间的距离（例子：大框包小框）

如何实现多类的NMS
每个类别自己的内部做NMS

NMS有那些改进方法？及改进思路

soft-nms：可以不要那么暴力的删除所有IOU大于阈值的框，而是降低其置信度
softer-nms：传统的nms用到的score仅仅是分类置信度得分，不能反应预测框的定位精度，也就是分类置信度和定位置信度非正相关。softer-nms基于soft-nms，对预测标注方差范围内的候选框进行加权平均，使得高定位置信度的预测框具有较高的分类置信度。预测的四个顶点坐标，分别对IOU>Nt的预测加权平均计算，得到新的4个坐标点。？？？
diou-nms：使用diou的方式计算iou。
adaptive-nms：自适应调整nms阈值，当检测目标不密集时，就使用较低的nms阈值，当检测目标密集出现许多重叠时，就是用较高的nms阈值。
weighted-nms：加权非极大值抑制，nms每次迭代所选出的最大得分框未必是精确定位的，冗余框也可能是良好的。weighted-nms与nms相比，就是在过滤矩形框时，不是直接采用IOU大于阈值，且类别相同的方法，而是根据网络预测的置信度进行加权，来得到新的预测矩形框。

# Code
我想这应该是最基础的部分，就是很有可能在面试的过程中，面试官就会要求手写一个nms的函数。所以这一部分就很重要。
这里的代码我仔细的看了一遍，我可以给一些参考方便理解，具体可以参考代码中的注释，注释写的还是很详细的。可以看出来每个框的shape都是[xmin,ymin,xmax,ymax,scores]，第一步取出每个的框的左上右下的xy值和score，每个值构成一个列表，接下来需要计算每个框的面积放在area中。然后按照score进行排序，argsort()是进行排序返回序号的函数。index中存放的是按score排序的每个框的序号值，接下来取出成绩最好的框，然后就是计算iou。这里这个参考的作者的注释有一点问题，就是当两个框不相交的时候，并不是x22-x11和y
22-y11为负数，而是x22-x11或者y22-y11有负数。不过这并不影响后面overlap的计算。下一个难理解的地方就是我们在计算了iou和比较完阈值之后，得到idx。这个idx实际上就是nms根据score最高的框没有剔除掉的预测框。那么这些框会进一步进行nms操作，直到index.size为0。还有人问为什么最后赋值index = index[idx + 1]。这里为什么要加一。是因为我们在计算时，将序号0这个框作为基准，也就是在计算x11，overlap，iou，idx时，都是从序号1这个框开始的。所以要+1。
```ruby
import numpy as np

import pdb
boxes = np.array([[100, 100, 210, 210, 0.72],
                  [250, 250, 420, 420, 0.8],
                  [220, 220, 320, 330, 0.92],
                  [100, 100, 210, 210, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [220, 230, 315, 340, 0.9]])

def py_cpu_nms(dets, thresh):
    # 首先数据赋值和计算对应矩形框的面积
    # dets的数据格式是dets[[xmin,ymin,xmax,ymax,scores]....]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    scores = dets[:, 4]
    print('areas  ', areas)
    print('scores ', scores)
    # 这边的keep用于存放，NMS后剩余的方框
    keep = []
    # 取出分数从大到小排列的索引。.argsort()是从小到大排列，[::-1]是列表头和尾颠倒一下。
    index = scores.argsort()[::-1]
    print('index:',index)
    # 上面这两句比如分数[0.72 0.8  0.92 0.72 0.81 0.9 ]
    #  对应的索引index[  2   5    4     1    3   0  ]记住是取出索引，scores列表没变。
    # index会剔除遍历过的方框，和合并过的方框。
    while index.size > 0:
        print(index.size)
        # 取出第一个方框进行和其他方框比对，看有没有可以合并的
        i = index[0]  # 取出第一个索引号，这里第一个是【2】
        # 因为我们这边分数已经按从大到小排列了。
        # 所以如果有合并存在，也是保留分数最高的这个，也就是我们现在那个这个
        # keep保留的是索引值，不是具体的分数。
        keep.append(i)
        print('keep:',keep)
        print('x1:', x1[i])
        print(x1[index[1:]])
        # 计算交集的左上角和右下角
        # 这里要注意，比如x1[i]这个方框的左上角x和所有其他的方框的左上角x的
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        print('*'*20)
        print(x11, y11, x22, y22)
        # 这边要注意，如果两个方框相交，X22-X11和Y22-Y11是正的。
        # 如果两个方框不相交，X22-X11和Y22-Y11是负的，我们把不相交的W和H设为0.
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        # 计算重叠面积就是上面说的交集面积。不相交因为W和H都是0，所以不相交面积为0
        overlaps = w * h
        print('overlaps is', overlaps)
        # 这个就是IOU公式（交并比）。
        # 得出来的ious是一个列表，里面拥有当前方框和其他所有方框的IOU结果。
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        print('ious is', ious)
        print(type(ious))
        # 接下来是合并重叠度最大的方框，也就是合并ious中值大于thresh的方框
        # 我们合并的操作就是把他们剔除，因为我们合并这些方框只保留下分数最高的。
        # 我们经过排序当前我们操作的方框就是分数最高的，所以我们剔除其他和当前重叠度最高的方框
        # 这里np.where(ious<=thresh)[0]是一个固定写法。
        idx = np.where(ious <= thresh)[0]
        print('idx:',idx)
        print(type(idx))
        # 把留下来框在进行NMS操作
        # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于thresh的框
        # 每一次都会先去除当前操作框，所以索引的列表就会向前移动移位，要还原就+1，向后移动一位
        # pdb.set_trace()
        index = index[idx + 1]  # because index start from 1
        print(index)
    return keep
keep = py_cpu_nms(boxes,0.7)
print(keep)
#参考：http://t.csdn.cn/rfIcN
```