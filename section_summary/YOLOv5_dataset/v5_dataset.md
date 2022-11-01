# YOLOv5---dataset
#### 代码声明：
本代码不是YOLOv5的官方代码，是参考b站的一个博主的。
参考链接：【Pytorch 搭建自己的YoloV5目标检测平台（Bubbliiiing 源码详解 训练 预测）-YoloV5整体结构介绍】 https://www.bilibili.com/video/BV1FZ4y1m777?share_source=copy_web&vd_source=7af49fc7f4527a070bbd7df512e8c736
为什么采用这个代码，首先就是我想对官方代码吐槽的，不可否认官方代码很工程化，就是我之前尝试阅读，发现它确实包含了各种情况，但是也正是因为这个原因，代码过于的工程化，并且注释较少，对于我这样的初学者实在是有点难，所以我选择了另一套代码，同时这个博主的代码实际上本身会有很多注释，也方便快速理解。
## 正文
接下来我们直接看这个代码中的dataset代码，首先我们先概括一下这个代码中包含什么，也就是本篇文章中涉及的内容。主要涉及两方面内容：dataset数据的提取、数据的增强，其中数据增强又包括mosaic，mixup，还有一些图像随机的缩放，色域变换，图像翻转以及各种数据增强之后bbox的变化策略。之前我也一直想要去看这一部分是怎么做的，这次终于搞定了。
```ruby
class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return image, box, y_true

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        return image_data, box#最后这里返回的是处理后的image和变换过的box==>（x,y,x,y）
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        return new_image, new_boxes
    
    def get_near_points(self, x, y, i, j):
        sub_x = x - i#相对于grid序号的偏差值
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]
        #为什么会根据偏差值返回不同的数组

    def get_target(self, targets):
        return y_true
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    y_trues = [[] for _ in batch[0][2]]
    for img, box, y_true in batch:
        images.append(img)
        bboxes.append(box)
        for i, sub_y_true in enumerate(y_true):
            y_trues[i].append(sub_y_true)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
    return images, bboxes,y_trues
```
这是整个dataset的代码，确实很长。而且不知道大家有没有同感，就是感觉深度学习的最难的永远是数据的读取。接下来呢，我们就一个模块一个模块的看。首先是进入datasets，这里我们需要清楚进入这个函数时的输入是什么，下面是我自己写的一个将dataset实例化，并从中提取数据的代码。
```ruby
if __name__ =='__main__':
    train_annotation_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/2007_train.txt'
    with open(train_annotation_path, encoding='utf-8') as f:#这里面存放的是图片的地址和bbox的标签---训练集
        train_lines = f.readlines()
    input_shape = [640,640]#输入的尺寸
    classes_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/model_data/loadmark_classes.txt'#类别文件存放的地址
    anchors_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/model_data/yolo_anchors.txt'#anchor大小文件存放的地址
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]#网络输出分为了三层，每层有三个anchor
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)
    UnFreeze_Epoch      = 50#总共进行的epoch
    mosaic              = True#是否进行mosaic
    mosaic_prob         = 0.5#进行mosaic的概率
    mixup               = True#是否进行mixup
    mixup_prob          = 0.5#进行mixup的概率
    special_aug_ratio   = 0.7#进行数据增强的epoch占总epoch的比例
    data_load1 = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, \
                mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
    pdb.set_trace()
    data = data_load1[0]
    print(len(data))
    print(data[0].size)
    print(data[1].size)
    print(data[2][0].shape)
    print(data[2][1].shape)
    print(data[2][2].shape)
```
首选进入代码中，我们设定好参数之后，会先去初始化，说白了就是把输入的参数放进类参数中。我们直接快进到getitem中。整体来看getitem分为几个部分：1、mosaic 2、mixup 3、数据随机增强 4、box-->y_true将真实框的(xywh类别)转换为神经网络的输出格式。
我们将分别介绍。

#### mosaic
首先进入getitem中，会得到index，接下来判断是否进行mosaic数据增强，判断条件就是考虑随机数，考虑epoch大小。
```ruby
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)#从所有的训练图片中抽取3张图片
            lines.append(self.annotation_lines[index])#将需要提取的图片加入到列表中
            shuffle(lines)#随机排列====目前是四张图片
            image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
```
这里有一个self.rand()函数，就是这个函数默认会有两个输入：a,b。最后这个函数return的是一个从a到b的随机数。默认a=0，b=1
```ruby
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
```
接下来就是mosaic的正文了，首先我们要了解其操作思想：将四张图片进行裁剪拼接放置于一张图片中。在这个叫做get_random_data_with_Mosaic的函数中，我们其实也可以将各个部分拆解。
```ruby
    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
```
到这里，实际上就是对将输入的annotation_line拆分开。annotation_line的形式是：```/home/ktd/rpf_ws/yolov5-pytorch-main/VOCdevkit/VOC2007/JPEGImages/37a.jpg 1309,724,1419,757,4 1070,725,1135,757,4```所以按空格切分形成列表，之后读取图片，将box放置在数组中。
```ruby
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]
```
这里就是对图片进行翻转，调用一个PIL.Image中的函数即可。而因为对数据进行翻转操作之后呢，需要对真实框相应的进行翻转操作。首先我们要清楚，这里是左右翻转，那么实际上y值是不变的。关于x呢又存在一个数学关系：翻转之后的x+翻转之前的x==图片x方向的长度w。这个地方想不明白可以画一画。
```ruby
            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)#缩放后新的宽高比
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
```
这个地方，是对图片的随机缩放，我感觉我甚至可以将这个地方理解为yolov5的多分辨率训练。因为这里jitter取0.3，所以宽高将会在[0.7,1.3]倍之间缩放，最终new_ar得到的是新的宽高比。而确定图片真正的尺寸是要通过scale进行缩放的。
```ruby
            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)
```
这里就是创建拼接图的第一步，首先创建一张灰度图，然后将每张图片贴在上面，但是会有一个偏移，在代码中的表示就是：图片的粘贴是通过左上角的坐标来做的，dx、dy就是左上角坐标。图张图拼接会有一个点作为交接点，这个点就是w * min_offset_x，h * min_offset_y。四张图对应四个序号0，1，2，3，他们按照逆时针顺序排列。
```ruby

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)
```
这一部分，主要是对bbox的计算，因为我们对图片进行了增强，那么bbox也会发生改变。很简单，因为之前对图片的翻转我们已经将bbox纠正过了，那么后面进行的图片缩放和dx，dy的粘贴的bbox矫正。实际上图片缩放后，bbox的坐标也就缩放多少。之后呢实际上就是对图片进行一个在偏移点的粘贴，所以bbox的xy也都加上dx，dy即可。但是需要注意就是bbox的取值范围。还需要注意一点就是这里image_datas是一个列表，里面存放的是四张粘贴过的照片。bbox_datas也是列表，里面存放的是每个照片的bbox，格式是Xmin，Ymin，Xmax，Ymax，c类别序号。
```ruby

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)

```
这里就是mosaic的最后一步了，也就是将image_datas里面的四张图片经过裁剪粘贴到同一张图上。首先还是找到分割点，即四张图片的交点。将这些图片都放在同一张图片中。
```ruby
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
```
这一部分是对图像进行色域变换。这里可能就需要简单了解一下图片的hsv是什么。
##### hsv
参考：http://t.csdn.cn/7RDQP
HSV(Hue, Saturation, Value)是根据颜色的直观特性由A. R. Smith在1978年创建的一种颜色空间, 也称六角锥体模型(Hexcone Model)（参考百度）。在HSV模型中，颜色是由色度（Hue），饱和度（Saturation），明度（Value）共同组成。
色度（Hue）使用角度度量的，范围是从0 ° 0\degree0°到360 ° 360\degree360°（逆时针旋转），比如0 ° / 360 ° 0\degree/360\degree0°/360°代表红色，120 ° 120\degree120°代表绿色，240 ° 240\degree240°代表蓝色。
饱和度（Saturation）表示颜色接近光谱色的程度。一种颜色，可以看成是某种光谱色与白色混合的结果。其中光谱色所占的比例愈大，颜色接近光谱色的程度就愈高，颜色的饱和度也就愈高（参考百度）。其范围是0到1。
明度（Value）颜色明亮的程度，对于光源色，明度值与发光体的光亮度有关。其范围是0（暗）到1（明）。
至于这里的hsv的随机数据增强，我不太理解。可能是对hsv这边还是不够理解，建议将这一块记成一个部分。

```ruby
        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox
```
这里的输入bbox_datas是存放了四张照片的bbox的列表。然后提取每个box，确认其在对应的图片区域。最后合并。这里输出的merge_bbox是一个列表中存放的每个bbox的xyxyc。
```ruby

        return new_image, new_boxes
```
到这里就是整个mosaic函数。

### mixup
首先我们要知道mixup的思想是什么：将两张图片按比例重叠，那么标签值也是。
但是我不明白这套代码的mixup是在mosaic之后的，也就是在做完mosaic之后，对其输出进行mixup的数据增强。不过没有关系，我们来仔细看mixup
首先它也是会从数据集中随意抽取出一个样本，之后。。。。我觉着到这里可以先看看第三部分，图片的随机增强。
欢迎你回来，因为针对mixup需要两张图片，将其按照一定透明度进行重叠。所以在代码中我们使用的两张图片分别是mosaic的输出和随意抽取的一张图像。那么针对随机抽取的图像进行数据的随机增强。
```ruby
            if self.mixup and self.rand() < self.mixup_prob:
                lines           = sample(self.annotation_lines, 1)
                image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
```
随机取出的图片进行随机数据增强得到image_2和box_2。然后将其和mosaic的输出image和box输入到mixip函数中。
```ruby
    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
```
这个函数很简单，直接将图片转换为numpy相加即可，bbox则是直接进行concat拼接即可。

### get random data
关于图片的随机增强就是这个函数get_random_data。这里面会有很多之前介绍过的东西，相信大家会很熟悉。
```ruby
    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()#这里将annotation_line切分为列表形式
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])#这里是说获取box的二维数组
```
到这里依然是根据index取出图片的地址打开图片，并且将对应的label转换为bbox数组的形式。
```ruby
        if not random:#这就是不做数据增强，图片和box的转变过程
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
```
上面这一部分，大家可能是不太熟悉，简单概括一下这一部分主要就是不进行数据增强时，将图片和label导出的代码。第一行的random，在类中取决与self.train。这也是整个模型是在训练阶段才进行数据增强的。这里在不进行数据增强后，首先计算图片的缩放前后的比例w/iw，取最小值，后面在用iw乘这个scale，实际上就是优先将最长的边进行缩放和目标值一样，再缩放短边，再补全短边。但是在实际过程中，长边缩放并不一定和目标值对齐，并且希望最后灰度条的补充是上下左右对称的，所以需要找到dx，dy两个点。那么后面真实框的调整就和之前一样，有缩放的比例，也有图片粘贴的位置dx，dy。那么问题也就迎刃而解了。
```ruby
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)#就这么一个函数实现图像翻转,左右的翻转

        image_data      = np.array(image, np.uint8)#PIL这个库读取图片之后不是np，所以转一步。
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:#首先搞清楚这里的box----->(Xmin,Ymin,Xmax,Ymax)
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx#*真实框的缩放比例+灰边的随机值==得到缩放后真实框的Xmin和Xmax的坐标
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy#同理，这里是y
            if flip: box[:, [0,2]] = w - box[:, [2,0]]#因为翻转只有左右翻转
            box[:, 0:2][box[:, 0:2]<0] = 0#这几行代码写的很神器，主要目的就是希望缩放之后的真实框始终在图中。
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]#根据坐标关系，得到box的wh
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] #这一步不太理解，感觉 就是单纯看这个预测框正不正确
        
        return image_data, box#最后这里返回的是处理后的image和变换过的box==>（x,y,x,y）
```
是不是感觉这一大部分都很熟悉，是不是感觉简单看看代码就知道是什么意思了，对没错，这一段就是我们之前在mosaic函数中用到的数据的随机增强，包括图片的缩放拉长，翻转，色域变换。
好啦，你可以回到第二部分啦，嘻嘻嘻。

### y_true！！！
其实我当初以为到这里，整个dataset就算结束了，但是并不是这样的。我们继续getitem看，会发现还有一个巨大的函数！
```ruby
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))#这里的preprocess_input是对图片进行归一化的处理
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            #这里的box已经是对图片操作之后，相对640，640的label值了
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]#这一步计算每个box的wh,这个wh相对于640的[0,1]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2#这一步计算每个box的中心点xy
        y_true = self.get_target(box)
        return image, box, y_true
```
首先我们清楚在做完mosaic、mixpu、或者get_random_data之后的image是(640,640,3)的图片格式，而box中label数据是Xmin，Ymin，Xmax，Ymax，c类别序号。第一行代码进行的是归一化，这里就不解释归一化的计算快等好处了。除了归一化，它还对数据的维度进行了调整将(640,640,3)-->(3,640,640),因为在神经网络模型中，我们的输出基本都是(bn,channels,w,h)。后面对box进行归一化！！！这个是重点，首先我们清楚在做完一些列的数据增量，box的值已经调整到了相对640的值。但是为什么归一化，为什么后面还会有一个get_target函数。这就需要充分了解yolo的理论思想。那就是将图片分成n * n的网格，只有物体中心所在的网格负责预测该物体，而这个物体的中心需要通过网格的序号以及相对该网格左上角的xy偏移量来表示。但是yolov5又有三个输出头：(默认输如640)步长分别为32，16，8，所以三层分别为20 * 20 ， 40 * 40 ， 80 *80 。所以要获得box中心点所在的网格序号，使用归一化后的xy成grid数即可，这是很方便的。同样wh也是，归一化之后再乘grid数后就可以获得该框在不同层的wh。因为在get_target函数中我们需要计算一个比例，来确定这个物体是否是这一层的anchor
负责检测。这就体现了yolo的另一个设定，那就是小特征图的感受野大，所以负责预测大物体，大特征图负责预测小物体。其实还不算完，get_target这个函数还要做一件事，那就是将box的数组形式转换为网络输出特征图的形式。举个例子：在20 * 20的特征图中，序号5 * 5的那个grid中有一个物体的中心点，偏移量dx，dy，box的wh分别为wh类别序号为c。那么实际上模型真实值应该为(bn,20,20,5 +类别个数),这里的输出就又是yolo的原理了，那么如果你不清楚或者比较模糊，建议看论文。这里我们可以先不考虑bn，假设类别个数为3，也就是在(20,20,8)的特征图中，我们需要将维度(5,5,0)设置为dx,(5,5,1)设置为dy，(5,5,2)设置为框宽w，(5,5,3)设置为框高h，(5,5,4)设置为置信度1，(5,5,5:)后面是类别的概率，采用one-hot编码。下面是代码，里面有一些我的注释。
```ruby
    def get_target(self, targets):
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]#这里num_layers只有三层，不清楚为什么会有四个键值对，每个键值对实际上对应一个步长[20,20]
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]#这里是构造了yolo模型真实输出的预测形式[3,20,20,16]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]#[3,20,20]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w      = grid_shapes[l]
            anchors         = np.array(self.anchors) / {0:32, 1:16, 2:8, 3:4}[l]#l对应层数，每一层对应一个步长。每个anchor都是相对于640图片的
            #所以这里需要除以步长，也化成网格形式
            
            batch_target = np.zeros_like(targets)#构建和targets相同形式的数组。此时在targets中，存放是的xywh都是相对于640的[0,1]的值
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w#之前已经化成了[0,1]所以要转换为网格数为多少，乘就完了。
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]
            #-------------------------------------------------------#
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            #-------------------------------------------------------#
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)#计算过后[4,9,2]
            #这个比率为什么这么算，还要扩展维度！我建议可以看看数组运算的广播机制！真的很神奇
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)#[4,9,2]
            ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)#[4,9,4]
            max_ratios           = np.max(ratios, axis = -1)#[4,9]
            
            for t, ratio in enumerate(max_ratios):
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True#np.argmin()返回的是最小值的序号
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))#batch_target是所有box按照grit划分后的值,t--->哪一个box框,i---->框中心点x属于哪个网格
                    j = int(np.floor(batch_target[t, 1]))
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)#batch_target[t, 0], batch_target[t, 1]中心点xy坐标，ij是中心点所在网格的序号
                    #返回了数组[0,0],[1,0],[0,1]
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:#确保这个点在grid 网格中
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:#这里不太清楚，在前面新设定的数组都是0，为什么还要检测是不是0。
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])#这里得到类别的序号。

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]#这里将比例也放在了存了进来？？不理解===#看上面，这就是为什么会比较box_best_ratio
                        
        return y_true
```
##### 如果你还有什么问题，欢迎来讨论。