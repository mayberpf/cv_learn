# Pytorch 模型推理及多任务通用范式
因为最近在做多任务的模型，所以准备详细的看一下这一块的实现。本篇文章是基于江大白ai实战课的笔记。可以通过微信公众号的搜索江大白看到。
## 安装
这几行代码我觉着会经常用到！！在经历了显卡驱动突然卡没了这件事之后，深刻领会到配置环境是多么让人绝望的事情。后面这几条代码相信大家肯定会用的到的。
```ruby
>>> import torch
>>> import torchvision
>>> print(torchvision.__version__)
0.13.1
>>> print(torch.cuda.is_available())
True
>>> print(torch.__version__)
1.12.1

```
## 模型载入
在课程中，示例使用的mobilenet_v2的模型，从本地加载模型。将mobilenet_v2进行实例化，就可以得到model，这里的model定义了每个网络层及他们的顺序。然后我们可以使用netron进行模型的可视化！！！
这个netron的安装，真实绝了，可能是因为我这里的网络不太好的原因。使用很多次pip安装都超时了。
```ruby
pip install netron #网络不好一直超时
snap install netron# 使用这种方法就成功了 
#在GitHub上找到源码，然后下载.APPImage，但是我下载后打不开
```
安装成功之后，我尝试了一下，加载个模型，但是我发现有一点和我预想的差别，那就是为什么我在加载自己模型之后，不像是官方模型那么好看，我自己的模型甚至连箭头都没有。后面在说。先继续往下进行。

按照课程中，我们在实例化模型之后，就得到了model。model.state_dict()是一个字典，它的key是网络层的名称(string格式)，对应的value是该层的参数(torch.tensor格式)。
这里我们只是加载了模型的结构，因此在加载完结构之后，模型各个网络层的参数是随机。但是如果我们加载预训练之后，每个网络层数的参数就会固定。在加载预训练权重
```ruby
model.load_state_dict(pretrained_state_dict(),strict = True)
```
pretrained_state_dict()这里存放的时预训练权重的参数。
### 从torchvision加载模型结构
```ruby
>>> import torch
>>> import torchvision.models as model
>>> mobilenet_v2 = model.mobilenet_v2()
>>> print(mobilenet_v2)
```
加载模型实际上就是这么简单，使用torchvision，当然这里没有加载权重参数，如果需要加载预训练模型的话，就需要本地有pth权重文件。至于torchvision的库中有什么模型，这个可以参考官方的文档。
```ruby
pretrained_state_dict  = torch.load('path of .pth')
model.load_state_dict(pretrained_state_dict,strict = True)
```
### 统计模型的参数量和计算量
关于这一块，需要一定的神经网络基础，我们可以举个例子：mobilenetv2模型第一层卷积，将三通道的图片转变成32通道数的特征图的卷积核为3*3的卷积这一步的参数量。首先我们可以根据模型加载之后，获得该层参数的shape值。通过下面代码，可以确定，模型的第一层参数也就是卷积层的参数的shape为(32,3,3,3)。那么就很容易理解了。这一层需要32 * 3 * 3 * 3=864个单精度数值，这个叫参数量。接下来我们来计算一下计算量。
这一层卷积将输入为(1,3,224,224)计算得到(1,32,112,112)的输出，其中卷积的步长为stride = 2，padding = 1。所以该层的计算量为
3(输入通道数)* 3*3(卷积核大小) * 224/2(图像的宽高进行卷积次数) * 224/2(图像宽高进行卷积次数) * 32(输出维度) = 10838016次单精度浮点运算(单位为：FLOPs)，一般为记为0.010838016 GFLOPs

```ruby
import torch
import torchvision.models as model
mobilenet_v2 = model.mobilenet_v2()
model_state_dict = mobilenet_v2.state_dict()
for key,value in model_state_dict.items():
    print(key)
    print(value.shape)
    break
# print(torch.__version__)
# print(torch.cuda.is_available())
```
针对整个模型的参数量和计算量我们可以通过使用Python现成的包thop来计算
```ruby
pip install thop#安装方法
inputs = torch.ones([1,3,224,224])
from thop import profile
flops , params = profile(model=mobilenet_v2,inputs=(inputs,))
print(flops,params)
```

### 将模型的结构和权重保存成onnx
一般将训练好的模型(结构和权重)转成onnx给到工程组
第一步加载模型结构，从本地和torchvision都可以
第二步读取权重和载入权重
第三步将模型放在cpu上(如果没有cuda的话)
第四步让模型变为推理状态
    这里将模型变为推理状态是因为：dropout和batchnorm层在训练状态和推理状态的处理方式是不同的。
第五步是构建一个项目推理时需要的输入大小的单精度tensor，并且放在对应的设备(cpu或cuda)
第六步是生成onnx
```ruby
import torch
import torchvision.models as model
mobilenet_v2 = model.mobilenet_v2()
inputs_ = torch.ones([1,3,224,224])
mobilenet_v2 = mobilenet_v2.to('cuda')
mobilenet_v2.eval()
inputs_ = inputs_.to('cuda')
# import pdb;pdb.set_trace()
torch.onnx.export(model=mobilenet_v2,args = inputs_,f ="./img/mobilenet_v2.onnx")
```
然后就可将产生的onnx文件放在netron中，就可以看到令人满意的网络可视化架构。所以说，如果你往netron中放入的是pth文件，那么就很难得到理想的全部的网络架构，因此建议多写几行代码，产生onnx文件，然后将起可视化即可。

## 构建模型
我们可以将模型的推理代码写进一个class中，该class的主流程predict方法永远只有三行代码
```ruby
inputs = self.preprocess(image)#数据预处理
outputs = self.model(inputs)#数据模型处理
results = self.postprocess(outputs)#数据后处理
```
数据预处理：神经网络模型的输入必须是满足要求的张量，所以我们需要对图片，语音，文字进行处理，将其转换为可以输入进神经网络的张量。
数据后处理：举个例子，目标检测模型输出的结构只是一个张量，其中包含了类别概率、置信度、box的xywh。我们需要经过后处理将box进行过滤，最后将确定的框画在图片上显示出来。

## 图片分类
我们按照三大步骤对推理代码进行编辑---这三大部分实际上可以写在一个类里面。
预处理
我们将需要预测的图片的使用cv2.imread进行读取，由于该读取方式将图片读取为BGR的numpy_array。所以我们需要将图片做简单的操作：BGR--->RGB;归一化；格式变化。代码简单可以表示为
```ruby
image = cv2.imread('the path of image')
image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image = cv2.resize(image,dsize = )
inputs = image/255
inputs = (inputs-np.array([0.485,0.456,0.406]))/np.array([0.229,0.224,0.225])
inputs = inputs.transpose(2,0,1)
inputs = inputs[np.newaxis,:,:,:]
inputs = torch.from_numpy(inputs)
inputs = inputs.type(float32)
inputs = inputs.to('cuda')
```
数据进入网络模型
这一部分除了简单的将经过预处理的图片送进模型，我们还需要做的就是将模型的权重参数加载进来。这一部分可以单独写一个函数
```ruby
def get_model(self):
    #上一节课的内容
    model = models.mobilenet_v2(num_classes=1000)
    pretrained_state_dict=torch.load('./weights/mobilenet_v2-b0353104.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(pretrained_state_dict, strict=True)
    model.to(self.device)
    model.eval()
    return model
```
在这个函数里，首先使用torchvision.models调佣mobilenet_v2的模型，然后加载模型权重。在加载权重时，有一个参数设置是```map_location =  lambda storage, loc: storage```这一步实际上就是在加载模型权重时，将权重先放回到cpu上。再读取。因为有时候别人保存权重文件时，可能并不会放在cpu中，所以在读取时就会导致torch.load报错。
接下来就是简单的图片进入模型
```ruby
outputs = self.model(inputs)
```
后处理
在进过模型处理之后，我们得到的outputs实际上还是一个tensor，数据后处理就是将这个tensor转变为和任务相关的label。举个例子，分类1000类别的图片，我们最后得到tensor的shape为(1,1000)，但是这个张量并没有经过sigmoid或者softmax。
这里需要注意：
```ruby
torch.nn.CrossEntropyLoss#自带softmax
torch.nn.BCEWithLogitsLoss#自带sigmoid
torch.nn.functional.binary_cross_entropy_with_logits()#自带sigmoid
```
于是我们可以根据tensor和label来书写postprocess代码了
```ruby
def postprocess(self, outputs):
    #取softmax得到每个类别的置信度
    outputs=torch.softmax(outputs,dim=1)
    #取最高置信度的类别和分数
    score, label_id = torch.max(outputs, dim=1)
    #Tensor ——> float
    score, label_id =score.item(), label_id.item()
    #查找标签名称
    label_name=self.label_names[label_id]
    return label_name, score
```
## 语义分割
这里选择使用的模型是DeepLabV3，在VOC数据集上做的预训练，其中包括20+1个类别，其中+1是指背景。
模型特点：
1、使用空洞卷积取代部分stride=2的下采样卷积，从而在不改变feature map宽高的前提下，来扩大感受野。（这一类型的操作，我前段时间也做了，但是效果好像不好。看来要再试试！）
2、Head结构中的ASPP模块，以不同的空洞采样率并行采样，同时提取不同感受野下的特征，功能类似于多尺度融合。
3、引入auxiliary block，为后续提供辅助Loss(只在训练时需要，推理时可以无视。)
接下来就是简单的推理部分，很简单，还是分三部分：预处理，模型处理，后处理。不同类型任务之间，预处理和模型处理基本相同，区别较大只有数据的后处理部分。因此我们这里直接将推理部分的类放进来，正好也可以看下框架。
```ruby
class ModelPipline(object):
    def __init__(self):
        #进入模型的图片大小：为数据预处理和后处理做准备
        self.inputs_size=(520,520)
        #CPU or CUDA：为数据预处理和模型加载做准备
        self.device=torch.device('cpu')
        #载入模型结构和模型权重
        self.model=self.get_model()
        #标签中的人体index，为数据后处理做准备
        self.person_id=15

    def predict(self, image):
        #数据预处理
        inputs, image_h, image_w=self.preprocess(image)
        #数据进网路
        outputs=self.model(inputs)
        #数据后处理
        results=self.postprocess(outputs, image_h, image_w)
        return results

    def get_model(self):
        #上一节课的内容
        model = models.segmentation.deeplabv3_resnet50(num_classes=21, pretrained_backbone=False, aux_loss=True)
        pretrained_state_dict=torch.load('./weights/deeplabv3_resnet50_coco-cd0a2569.pth', map_location=lambda storage, loc: storage)
        model.load_state_dict(pretrained_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image):
        #opencv默认读入是BGR，需要转为RGB，和训练时保持一致
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        #提取原图大小
        image_h, image_w=image.shape[:2]
        #resize成模型输入的大小，和训练时保持一致
        image=cv2.resize(image, dsize=self.inputs_size)
        #归一化和标准化，和训练时保持一致
        inputs=image/255
        inputs=(inputs-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        ##以下是图像任务的通用处理
        #(H,W,C) ——> (C,H,W)
        inputs=inputs.transpose(2,0,1)
        #(C,H,W) ——> (1,C,H,W)
        inputs=inputs[np.newaxis,:,:,:]
        #NumpyArray ——> Tensor
        inputs=torch.from_numpy(inputs)
        #dtype float32
        inputs=inputs.type(torch.float32)
        #与self.model放在相同硬件上
        inputs=inputs.to(self.device)
        return inputs, image_h, image_w

    def postprocess(self, outputs, image_h, image_w):
        #获取模型输出output
        outputs=outputs['out']
        #取softmax得到每个类别的置信度
        outputs=torch.softmax(outputs,dim=1)
        #取出目标标签（比如：人体）的那一层置信度
        outputs=outputs[:,self.person_id:self.person_id+1,:,:]
        #将结果图resize回原图大小
        outputs=F.interpolate(outputs, size=(image_h, image_w), mode='bilinear',align_corners=True)
        #数据类型转换：torch.autograd.Variable ——> torch.Tensor ——> numpy.ndarray
        mask_person=outputs.data.cpu().numpy().squeeze()
        return mask_person
```
接下来我们简单说一下后处理的部分，在这个例子中我们使用的是deeplabv3。对输入数据进行分类，一共还是拥有20+1个类别。所以我们模型的输出shape为（1，21，520，520），这里1表示batch_size。这里的21表示的是21个类别，也可以理解为通道数，那就是每个通道负责预测一个类别，也就是每个通道都是一个01分类。520，520就是模型输出特征图的宽高。
接下来就是整体类的调用，然后进行推理。
```ruby
if __name__=='__main__':
    #实例化
    model_segment=ModelPipline()

    #第一张图
    image=cv2.imread('./images/person_0.jpg')
    result=model_segment.predict(image)
    #可视化结果
    cv2.imwrite('./demos/person_0_mask.jpg',(result*255).astype(np.uint8))
    mask=result.copy()
    mask[mask>=0.5]=255
    mask[mask<0.5]=0
    image_mask=np.concatenate([image,mask[:,:,np.newaxis]],axis=2)#这里的拼接完之后的图片的通道数为4
    cv2.imwrite('./demos/person_0_mask.png',image_mask)#但是这里显示的就是rgb图片在mask掩码下的图片。

    #第二张图
    image=cv2.imread('./images/person_1.jpg')
    result=model_segment.predict(image)
    #可视化结果
    cv2.imwrite('./demos/person_1_mask.jpg',(result*255).astype(np.uint8))
    mask=result.copy()
    mask[mask>=0.5]=255
    mask[mask<0.5]=0
    image_mask=np.concatenate([image,mask[:,:,np.newaxis]],axis=2)
    cv2.imwrite('./demos/person_1_mask.png',image_mask)
```
可视化里面有一个地方，那就是保存通道数为4的图片那里。我试过了，只有保存图片才会产生抠图的效果。cv2.imshow()不会显示出那种效果。
## 目标检测
这里在进行目标检测和图像分割时，包括预处理和模型处理实际上和前面的基本没有很大的差别。
主要差别在于数据后处理。这里以YOLOX为例子
主要可以分为八个步骤：
1、将每个anchor的中心点横坐标、中心点纵坐标、框的宽高、转换为anchor的左上角和右下角坐标。
2、因为在推理过程中只是针对一张图片进行处理，所以将输出维度从(1,8400,85)--->(8400,85)
3、求出每个anchor分数最高的类别，将该分数和前景分数相乘，作为anchor最终的置信度，和人为设置的阈值进行比较，判断该anchor是否要保留
4、将anchor的位置信息、置信度信息、类别信息整合在一起
5、剔除置信度小于阈值的anchor框
6、如果此时所有的anchor都被剔除了，那么返回None
7、NMS非极大值抑制。
8、计算最终保留的anchor框在原图比例下的位置。
我们将整个代码显示出来：
```ruby
import torch
import torchvision
import numpy as np
import cv2
from models_yolox.yolox_s import YOLOX
from models_yolox.visualize import vis


class ModelPipline(object):
    def __init__(self):
        #进入模型的图片大小：为数据预处理和后处理做准备
        self.inputs_size=(640,640)#(h,w) 

        #CPU or CUDA：为数据预处理和模型加载做准备
        self.device=torch.device('cpu')

        #载入模型结构和模型权重
        self.num_classes=80
        self.model=self.get_model()

        #后处理的阈值
        self.conf_threshold=0.5
        self.nms_threshold=0.45

        #标签载入
        label_names=open('./labels/coco_label.txt','r').readlines()
        self.label_names=[line.strip('\n') for line in label_names]
        

    def predict(self, image):
        #数据预处理
        inputs, r=self.preprocess(image)
        #数据进网络
        outputs=self.model(inputs)
        #数据后处理
        results=self.postprocess(outputs, r)
        return results

    def get_model(self):
        #Lesson 2 的内容
        model = YOLOX(num_classes=self.num_classes)
        pretrained_state_dict=torch.load('./weights/yolox_s_coco.pth.tar', map_location=lambda storage, loc: storage)["model"]
        model.load_state_dict(pretrained_state_dict, strict=True)
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image):
        # 原图尺寸
        h, w = image.shape[:2]
        # 生成一张 w=h=640的mask，数值全是114
        padded_img = np.ones((self.inputs_size[0], self.inputs_size[1], 3)) * 114.0
        # 计算原图的长边缩放到640所需要的比例
        r = min(self.inputs_size[0] / h, self.inputs_size[1] / w)
        # 对原图做等比例缩放，使得长边=640
        resized_img = cv2.resize(image, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        # 将缩放后的原图填充到 640×640的mask的左上方
        padded_img[: int(h * r), : int(w * r)] = resized_img
        # BGR——>RGB
        padded_img = padded_img[:, :, ::-1]
        #归一化和标准化，和训练时保持一致
        inputs=padded_img/255
        inputs=(inputs-np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])
        ##以下是图像任务的通用处理
        #(H,W,C) ——> (C,H,W)
        inputs=inputs.transpose(2,0,1)
        #(C,H,W) ——> (1,C,H,W)
        inputs=inputs[np.newaxis,:,:,:]
        #NumpyArray ——> Tensor
        inputs=torch.from_numpy(inputs)
        #dtype float32
        inputs=inputs.type(torch.float32)
        #与self.model放在相同硬件上
        inputs=inputs.to(self.device)
        return inputs, r

    def postprocess(self, prediction, r):
        #prediction.shape=[1,8400,85]，下面先将85中的前4列进行转换，从 xc,yc,w,h 变为 x0,y0,x1,y1
        box_corner = prediction.new(prediction.shape)
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]
        #只处理单张图
        image_pred=prediction[0]
        #class_conf.shape=[8400,1],求每个anchor在80个类别中的最高分数。class_pred.shape=[8400,1],每个anchor的label index。
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + self.num_classes], 1, keepdim=True)
        conf_score = image_pred[:, 4].unsqueeze(dim=1) * class_conf
        conf_mask = (conf_score >= self.conf_threshold).squeeze()
        #detections.shape=[8400,6]，分别是 x0 ,y0, x1, y1, obj_score*class_score, class_label 
        detections = torch.cat((image_pred[:, :4], conf_score, class_pred.float()), 1)
        #将obj_score*class_score > conf_thre 筛选出来
        detections = detections[conf_mask]
        #通过阈值筛选后，如果没有剩余目标则结束
        if not detections.size(0):
            return None
        #NMS
        nms_out_index = torchvision.ops.batched_nms(detections[:, :4],detections[:, 4],detections[:, 5],self.nms_threshold)
        detections = detections[nms_out_index]
        #把坐标映射回原图
        detections=detections.data.cpu().numpy()
        bboxes=(detections[:,:4] / r).astype(np.int64)
        scores=detections[:,4]
        labels=detections[:,5].astype(np.int64)
        return bboxes, scores, labels

        

if __name__=='__main__':
    model_detect=ModelPipline()
    label_names=model_detect.label_names

    image=cv2.imread('./images/1.jpg')
    result=model_detect.predict(image)
    if result is not None:
        bboxes, scores, labels = result
        image=vis(image, bboxes, scores, labels, label_names)
    cv2.imwrite('./demos/1.jpg',image)

    image=cv2.imread('./images/2.jpg')
    result=model_detect.predict(image)
    if result is not None:
        bboxes, scores, labels = result
        image=vis(image, bboxes, scores, labels, label_names)
    cv2.imwrite('./demos/2.jpg',image)

    image=cv2.imread('./images/3.jpg')
    result=model_detect.predict(image)
    if result is not None:
        bboxes, scores, labels = result
        image=vis(image, bboxes, scores, labels, label_names)
    cv2.imwrite('./demos/3.jpg',image)


```

