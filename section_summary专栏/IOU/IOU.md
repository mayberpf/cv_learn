# IOU总结
首先，我们知道，在目标检测中，主要目标是希望预测框bbox的角点和真实值无限接近。在IOU loss出现之前主要是采用L1 ，L2loss对这一目标进行优化。但是L1L2 loss分别作用在几个不同的目标上，并没有关联性。因此它的结果是很不精确的。本篇文章主要讲解与iou相关的一系列。
# IOU Loss
令人熟知，IOU起初就是交集比并集。那么IOU loss 实际上就是预测框和真实框之间交集比并集的负对数。IOU的取值范围在[0,1]，对应ln函数，IOU loss就可想而知。但是它存在一个最大的问题就是：当预测框和真实框之间没有交集时，无法为梯度下降算法提供梯度，也就是当没有交集时，无法判断预测框和真实框之间是比较接近还是距离很远。
下面的代码是在yolov5中使用到iou的计算过程。
```ruby
        #----------------------------------------------------#
        #   求出预测框左上角右下角
        #----------------------------------------------------#
        b1_xy       = b1[..., :2]
        b1_wh       = b1[..., 2:4]
        b1_wh_half  = b1_wh/2.
        b1_mins     = b1_xy - b1_wh_half
        b1_maxes    = b1_xy + b1_wh_half
        #----------------------------------------------------#
        #   求出真实框左上角右下角
        #----------------------------------------------------#
        b2_xy       = b2[..., :2]
        b2_wh       = b2[..., 2:4]
        b2_wh_half  = b2_wh/2.
        b2_mins     = b2_xy - b2_wh_half
        b2_maxes    = b2_xy + b2_wh_half

        #----------------------------------------------------#
        #   求真实框和预测框所有的iou
        #----------------------------------------------------#
        intersect_mins  = torch.max(b1_mins, b2_mins)
        intersect_maxes = torch.min(b1_maxes, b2_maxes)
        intersect_wh    = torch.max(intersect_maxes - intersect_mins, torch.zeros_like(intersect_maxes))
        intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
        b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
        b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
        union_area      = b1_area + b2_area - intersect_area
        iou             = intersect_area / union_area
```
# GIOU
GIOU主要解决的就是预测框和真实框之间没有交集的问题。也就是在原始IOU的基础上添加了一个惩罚项。首先GIOU定义了一个值：预测框和真实框的最小外包面积。这个惩罚项就是：最小外包面积-并集的面积之后再除以最小外包面积。
缺点：预测框被包含在了真实框中，这时最小包围面积和并集面积相同，那么GIOU的惩罚值为0。此时GIOU则退化成了IOU。同时，GIOU倾向于增大预测框的面积使其包含真实框来使得惩罚项为0，大大增加了收敛所需要的时间。
接着上面的代码，在计算了iou之后的GIOU的代码是：
```ruby
        #----------------------------------------------------#
        #   找到包裹两个框的最小框的左上角和右下角
        #----------------------------------------------------#
        enclose_mins    = torch.min(b1_mins, b2_mins)
        enclose_maxes   = torch.max(b1_maxes, b2_maxes)
        enclose_wh      = torch.max(enclose_maxes - enclose_mins, torch.zeros_like(intersect_maxes))
        #----------------------------------------------------#
        #   计算对角线距离
        #----------------------------------------------------#
        enclose_area    = enclose_wh[..., 0] * enclose_wh[..., 1]
        giou            = iou - (enclose_area - union_area) / enclose_area
```
# DIOU loss
上面说的GIOU确实还存在一个问题：当预测框和真实框并排放置，此时GIOU也为0，IOU也为0。
因此DIOU loss的公式：1-IOU+距离的平房比。这里因为不方便打公式，所以使用文字来说明，想要了解公式可以去网上查阅一下资料。
下面来解释一下这个距离比：在网上的资料中说，分子是预测框中心点和真实框中心点的欧氏距离的平方，分母是最小包围框的对角线的平方。
关于欧氏距离，和我们在二维平面说说两点之间的最短直线距离一样。
优点：与GIOU类似，在DIOU loss在与目标框不重叠时，仍然可以为边界框提供移动距离。
DIOU loss可以直接最小化两个目标框的距离，而GIOU优化的是两个目标框之间的面积，因此比GIOU收敛的快得多
对于包含两个框在水平方向和垂直方向上这种情况，DIOU损失可以使回归非常快，GIOU损失几乎退化为IOU损失
代码参考：http://t.csdn.cn/BoNZ2
```ruby
def bboxes_diou(boxes1,boxes2):
    '''
    cal DIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    #cal the box's area of boxes1 and boxess计算预测框和真实框的面积
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    #cal Intersection找到交集的左上角和右下角
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)#没有交集会出现负数，后面算面积，所以用0代替
    inter_area = inter_section[...,0] * inter_section[...,1]#计算交集面积
    union_area = boxes1Area+boxes2Area-inter_area#计算并集面积
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)#IOU的计算

    #cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])#最小外围包围框的左上角
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])#最小包围框的右下角
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)#
    outer_diagonal_line = np.square(outer[...,0]) + np.square(outer[...,1])
    #np.square是获得平方，这里获得的是对角线的平方

    #cal center distance
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5#计算中心点
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = np.square(boxes1_center[...,0]-boxes2_center[...,0]) +\
                 np.square(boxes1_center[...,1]-boxes2_center[...,1])
#计算两个中心点之间的距离也是平方
    #cal diou
    dious = ious - center_dis / outer_diagonal_line

    return dious

```

# CIOU loss
在一篇文章中，好的回归loss应该综合考虑目标框与真实框的重叠区域，中心点距离以及宽高比一致性这三个问题。其中IOU考虑了重叠面积，DIOU考虑了中心点距离、因此CIOU则是在DIOU的基础上添加了宽高比一致性loss。
也就是CIOU会比DIOU多出一个衡量宽高比的式子：αv.这里的α是用于平衡比例的参数。v是用来衡量anchor和目标框之间的比例一致性。

```ruby
def bboxes_ciou(boxes1,boxes2):
    '''
    cal CIOU of two boxes or batch boxes
    :param boxes1:[xmin,ymin,xmax,ymax] or
                [[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    :param boxes2:[xmin,ymin,xmax,ymax]
    :return:
    '''

    #cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[...,2]-boxes1[...,0])*(boxes1[...,3]-boxes1[...,1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # cal Intersection
    left_up = np.maximum(boxes1[...,:2],boxes2[...,:2])
    right_down = np.minimum(boxes1[...,2:],boxes2[...,2:])

    inter_section = np.maximum(right_down-left_up,0.0)
    inter_area = inter_section[...,0] * inter_section[...,1]
    union_area = boxes1Area+boxes2Area-inter_area
    ious = np.maximum(1.0*inter_area/union_area,np.finfo(np.float32).eps)

    # cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[...,0]) + np.square(outer[...,1])

    # cal center distance
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = np.square(boxes1_center[...,0]-boxes2_center[...,0]) +\
                 np.square(boxes1_center[...,1]-boxes2_center[...,1])

    # cal penalty term
    # cal width,height
    boxes1_size = np.maximum(boxes1[...,2:]-boxes1[...,:2],0.0)
    boxes2_size = np.maximum(boxes2[..., 2:] - boxes2[..., :2], 0.0)
    v = (4.0/np.square(np.pi)) * np.square((
            np.arctan((boxes1_size[...,0]/boxes1_size[...,1])) -
            np.arctan((boxes2_size[..., 0] / boxes2_size[..., 1])) ))
    alpha = v / (1-ious+v)
    #cal ciou
    cious = ious - (center_dis / outer_diagonal_line + alpha*v)

    return cious

```
# SIOU loss
SIOU不仅考虑了之前提到的重叠面积，中心距离，宽高比三个问题，在这个基础上还考虑了真实框与预测框之间构成的方向角度之间的差异。
也就是评估真实框预测框中心点连线和x，y轴之间的夹角关系。这里的α是指与x轴夹角。如果α大于45度，则想办法将其优化到90度，如果小于45度，则将其优化到0度。

```ruby
elif self.iou_type == 'siou':
	# SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
	'''
	预测框和真实框坐标形式为xyxy，即左下右上角坐标或左上右下角坐标
	'''
	s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 #真实框和预测框中心点的宽度差
	s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 #真实框和预测框中心点的高度差
	sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5) #真实框和预测框中心点的距离====torch.pow()实现指数操作
	sin_alpha_1 = torch.abs(s_cw) / sigma #真实框和预测框中心点的夹角β====这里应该是夹角对应的sin值
	sin_alpha_2 = torch.abs(s_ch) / sigma #真实框和预测框中心点的夹角α====这里应该是夹角对应的sin值
	threshold = pow(2, 0.5) / 2 #夹角阈值
	sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1) #α大于45°则考虑优化β，否则优化α
	angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2) #角度损失
	rho_x = (s_cw / cw) ** 2 
	rho_y = (s_ch / ch) ** 2
	gamma = angle_cost - 2
	distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y) #距离损失
	omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
	omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
	shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4) #形状损失
	iou = iou - 0.5 * (distance_cost + shape_cost) #siou

loss = 1.0 - iou
```