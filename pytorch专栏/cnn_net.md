#CNN_Net
要点：
1、自定义类要继承torch.nn.Module。有时自己设计了一些模块，为了方便使用，通常额外定义一个类，当然也需要继承nn.Module
2、完成init函数和forward函数，其中init函数完成网络的搭建，forward函数完成网络的前向传播
3、完成所有层的参数初始化，一般只有卷积层、归一化层、全连接层要初始化，池化层没有参数
####__init__函数
构建网络层有几种方式，一种是使用pytoch官方已经有了定义的网络：resnet，vgg,inception等一种是自定义层，例如自己设计了一个新的模块。
首先，pytorch官方库已经支持的网络，这些网络放在了torchvision.models中，下面选择自己需要的一个。
```r
import  torchvision.models as models
resnet18 = models.resnet18(pretrained = True) 
alexnet = models.alexnet()
```
若需要加载网络在imagenet上的预训练模型，设置pretrained参数为True即可。但是这种方式每次运行都会从网上读取预训练模型，所以可以去官网下载，然后本地加载
```r
resnet18.load_state_dict(torch.load('这里是预训练权重的地址'))
```
自定义层可以通过
```r
import torch.nn as nn
self.trunk_52 = nn.Sequential()
```
在其中可以添加卷积，归一化，激活函数，池化。

当网络模型很深时，就不能够全部列出来，必须找到模型的共同部分，然后通过传参设置不同的层。
步骤：1、先构建一个由几层网络搭建的类，每个参数可以通过传参设置。2、设置不同的层。
（1的步骤我们较为了解，这里就只展示2的代码例子）
```r
layers = []
layers.append(block(self.inplanes,stride,downsample,self.groups,self.basewidth,previousdilation,norm_layer))
self.inplanes = planes * block.expansion
for _ in range(1,blocks):
    layers.append(block(self.inplanes,stride,downsample,self.groups,self.basewidth,previousdilation,norm_layer))
return nn.Sequential(*layers)
```
####forward函数
这里就是前向传播，一般情况都是一路向下，你也可以自己封装一个类，来完成一些金字塔结构都是可以的，如果要可视化特征图，只需要return 该特征图即可
####初始化网络
网络模型初始化要放在init中，分为两类：一类是随机初始化，一类是加载预训练模型
有关随机初始化之前在net_init中提到过，所以这里重点介绍预训练模型的的加载。
#####加载预训练模型初始化
加载预训练模型一般都是在train文件中写，但是有些网络由于是使用现成的backbone网络，因此需要针对backbone加载预训练模型，其他层采用随机初始化的方法，这就需要在网络中定义了。
最简单的就是直接加载整个模型
网络的每一层是如何表示的
```r
model = net()
for name, value in model.named_parameters():
    print(name)
```
在预训练模型中，有key即为网络层的名字，value即为他们对应的参数。因此，在加载预训练模型可以按照下面的方式加载
```r
pretrained_dict = torch.load('地址')
pretrained_dict.pop('fc.weight')
pretrained_dict.pop('fc.bias')
model_dict = model.statedict()
pretrained_dict = {k:v for k , v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
```