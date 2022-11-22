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
## 目标检测

