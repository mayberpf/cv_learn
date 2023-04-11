# NMS非极大值抑制
说实话，在这之前，我已经写过有关非极大值抑制的解析了。所以这里简单的说明一下。
NMS用来抑制检测时冗余的框。
流程：
对所有检测框的置信度进行排序-->用置信度最高的框为基准，与其他框计算IOU--->IOU大于阈值的框选择滤除掉。
直到所有的框都小于这个阈值。
当然，这里所说的所有框，是指一个类别的所有框。多类别就需要多执行几次。但是你也可以通过更改代码，使NMS对全部类别进行过滤。
代码：
```ruby
import torch
import cv2
import numpy as np
def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1) # 每个框的面积 (N,)
    area2 = box_area(boxes2) # (M,)
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    # [N,M,2] # N中一个和M个比较; 所以由N,M 个
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) # [N,M,2]
    wh = (rb - lt).clamp(min=0) # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    #小于0的为0
    # [N,M]
    iou = inter / (area1[:, None] + area2 - inter)
    return iou
    # NxM, boxes1中每个框和boxes2中每个框的IoU值;
def ori_nms(boxes, scores, iou_threshold):
"""
:param boxes: [N, 4], 此处传进来的框,是经过筛选(NMS之前选取过得分TopK)之后, 在传
入之前处理好的;
:param scores: [N]
:param iou_threshold: 0.7
:return:
"""
    keep = []
    # 最终保留的结果, 在boxes中对应的索引;
    idxs = scores.argsort() # 值从小到大的 索引
    while idxs.numel() > 0: # 循环直到null; numel(): 数组元素个数
        # 得分最大框对应的索引, 以及对应的坐标
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        # [1, 4]
        keep.append(max_score_index)
        if idxs.size(0) == 1:
        # 就剩余一个框了;
        break
        idxs = idxs[:-1]
        # 将得分最大框 从索引中删除; 剩余索引对应的框 和 得分最大框 计算IoU;
        other_boxes = boxes[idxs]
        # [?, 4]
        ious = box_iou(max_score_box, other_boxes)
        # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
    keep = idxs.new(keep)
    # Tensor
    return keep
```
可以自己画点框进行验证。
```ruby
img = np.zeros([224, 224, 3], np.uint8)
box1 = [20, 50, 125, 189]
box2 = [34, 55, 110, 201]
box4 = [78, 180, 157, 200]
box3 = [112, 131, 155, 195]
cv2.rectangle(img, (box1[0], box1[2]), (box1[1], box1[3]), (0, 255, 0), 2)
cv2.rectangle(img, (box2[0], box2[2]), (box2[1], box2[3]), (0, 255, 0), 2)
cv2.rectangle(img, (box3[0], box3[2]), (box3[1], box3[3]), (0, 255, 0), 2)
cv2.rectangle(img, (box4[0], box4[2]), (box4[1], box4[3]), (0, 255, 0), 2)
cv2.imwrite('test.jpg', img)
box1 = [20.0, 50.0, 125.0, 189.05]
box2 = [34.0, 55.0, 110.0, 201.0]
box4 = [78.0, 180.0, 157.0, 200.0]
box3 = [112.0, 131.0, 155.0, 195.0]
bbox =
torch.tensor([box1, box2, box3, box4])
score = torch.tensor([0.5, 0.3, 0.2, 0.4])
output = ori_nms(boxes=bbox, scores=score, iou_threshold=0.3)
print(output)
```
当然我们也可以直接调用
```ruby
from torchvision.ops import nms
keep = nms(boxes = bbox,scores = score,iou_threshold = 0.3)
print(keep)
```
关于NMS的优化，主要还是针对IOU进行的，所以我们看下，IOU的进化史。
## IOU及其进化
### IOU
IOU就是最简单的交集比并集！
IOU存在的缺点就是：如果两个框没有相交，IOU=0，这是不能反应两者之间的距离的。同时IOu无法精确反映两者的重合度大小。也就是存在IOU相同但是重合度不同的情况。
### GIOU
其取值为[-1,1]，两者完全重合时，取1，两者无交集且无限远时，取-1。所以GIOU实际上优化的是上面的第一个缺点。
GIOU的计算为：IOU减去两个框的闭包区域中，不属于两个框的区域所占的比例。
### DIOU
将目标与anchor之间的距离，重叠率以及尺度都考虑进去，使目标框回归变得更加稳定。DIOU是在GIOU的基础上有加上了一个惩罚，那就是两个框的中心点距离与闭包区域的对角线长度。
### CIOU
怎么说呢，在DIOU的基础上又考虑了长宽比！！！

总之，IOU的进化，就是疯狂加加加
```ruby
    def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False,eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:
    # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
    # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)*(torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)# Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)# convex(smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
    # convex height
    # Distance or Complete IoUhttps://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps
    # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +(b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
    # center distancesquared
            if DIoU:
                return iou - rho2 / c2
            elif CIoU:
    # DIoU
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) -torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
        else:
    # CIoU
    # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps
    # convex area
            return iou - (c_area - union) / c_area
    # GIoU
    else:
    return iou# IoU
```
总结一下：GIOU在IOU的基础上解决边界框不重合时的问题。DIOU在IOU和GIOU的基础上考虑了边界框中心点的信息。CIOU在DIOU的基础上，考虑了边界框宽高比的尺度信息。

## NMS的优化
这里简单的列举下：
soft-nms可以不暴力的删除IOU大于阈值的框，而是降低置信度
softer-nms加权平均
DIOU-nms使用diou计算
adaptive nms根据框的密集程度进行阈值的调整
weighted nms加权非极大值抑制。