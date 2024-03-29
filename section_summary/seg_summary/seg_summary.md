# 图像分割总结
分为三类：语义分割、实例分割、全景分割
之前的分割，比如数字图像处理的阈值分割，一般就是按照灰度值或者像素值分割，在分割完后还存在很多噪声，很多符合特征阈值内的非目标区域，一般都会做一下形态学处理---->之前写过的基本的图像处理，包括腐蚀膨胀之类的，就可以将大部分的小噪声或者其他小的干扰区域给去除
传统方法的泛化行不强，只能用于某些特定场合，不能够广泛应用。
Unet
典型的神经网络方法，将输入图片进行编码和解码。其中为了应对一些难以分割的区域，尺度变换大的目标等问题，还设计了对应的特征金字塔、注意力机制等相关技术模块。
### 难点
微小物体的分割和分类
看起来相似的物体，比如猫狗的分类
背景较为复杂的情况下，如何识别和分离背景
关于不同角度的问题，如何处理
### 数据集
最重要的就是：原始RGB图片和标注好的RGB图片
### 评价指标
像素准确度
平均交并比mIOU
平均像素准确度
频率加权交并比
### 编码器-解码器优缺点
优点：
能够提取高层次的特征信息:编码器通过卷积层等操作,能够有效地提取图像的高层次特征信息,这些信息对于图像分类、目标检测、语义分割等任务非常重要。
能够还原图像细节信息:解码器通过上采样、反卷积等操作,能够将编码器输出的低分辨率特征图还原为原始图像大小的高分辨率特征图,从而恢复出原始图像中的细节信息。
结构简单:编码器-解码器结构是一种相对简单的网络结构,易于理解和实现,并且具有较好的可解释性。
缺点：
由于多次卷积和池化操作,编码器-解码器结构会使得图像信息逐渐丢失,导致一些细节信息无法恢复,例如边缘和纹理等。
在解码器中使用反卷积和上采样等操作,容易引起信息的混叠和失真,导致图像质量下降。
编码器-解码器结构的训练需要大量的数据和计算资源,模型参数较多,训练过程较为困难。
### FCN
全连接层的使用，会破坏原有的空间信息，且不再具有局部信息，这对于图像分割这种像素级别的预测任务来说，印象非常大
整体的网络结构：
对整张图片进行逐层下采样，通道数增加的4096后，再通过1x1卷积降维到类别数量的大小，从而保证每个feature map都对应了一个类别，但论文中的图没有表示出来，降采样之后的feature map 如何回到原图的大小。但是看代码就很容易get到
```ruby
class FCN(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, **_):
        super(FCN, self).__init__()
        #self.relu = nn.ReLU(inplace=True)

        # VGG
        vgg = models.vgg16(pretrained=pretrained)
        encoder = list(vgg.features.children())

        self.pool3 = nn.Sequential(*encoder[0:17])

        self.pool4 = nn.Sequential(*encoder[17:24])

        self.pool5 = nn.Sequential(*encoder[24:31])

        self.conv_6 = nn.Sequential(nn.Conv2d(512, 4096, (7, 7), padding=1),nn.ReLU(inplace=True))

        self.conv_7 = nn.Sequential(nn.Conv2d(4096, 4096, (1, 1)),nn.ReLU(inplace=True))

        self.conv_8 = nn.Conv2d(4096, num_classes, (1, 1))

        self.convTrans_9 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convTrans_10 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv1x1_3_4 = nn.Conv2d(256, num_classes, (1, 1))

        self.conv1x1_4_4 = nn.Conv2d(512, num_classes, (1, 1))

        self.drop6 = nn.Dropout2d()

        self.drop7 = nn.Dropout2d()

        self.convTrans_11 = nn.UpsamplingBilinear2d(scale_factor=8)

        #self.final = nn.Sequential(nn.Conv2d(num_classes, num_classes, (1, 1)),nn.Softmax2d())

    def forward(self, x):
        x = x.float()
        x = self.pool3(x)
        f3 = self.conv1x1_3_4(x)
        x = self.pool4(x)
        f4 = self.conv1x1_4_4(x)
        x = self.pool5(x)
        x = self.conv_6(x)
        x = self.drop6(x)
        x = self.conv_7(x)
        x = self.drop7(x)
        x = self.conv_8(x)
        x = self.convTrans_9(x)
        x += f4
        x = self.convTrans_10(x)
        x += f3
        x = self.convTrans_11(x)

        output = x
        return output
```
### U-net
基于FCN架构，在具体细节上进行改进。
unet采用反卷积或者线性插值的方式获得上层特征，与网络前面部分裁剪到相同大小后再拼接
改进的考虑：
1、充分利用浅层特征的信息，因为unet解决的初衷是医学图像，背景较为单一，因此浅层特征是可以充分利用的，不同于自然图像，浅层这种存在大量复杂的背景，信噪比比较小，因此自然图像不适合这么做！！！
2、由于使用反卷积完成上采样，因此边缘区域实际是缺失的，通过使用浅层的特征可以很好的补充这些缺失区域的信息。
3、相比于相加的方式融合，拼接的方式使得特征更多，在通过卷积使得这些特征能够更好的融合。
网络的模型，直接看代码就能了解。
```ruby
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

### deeplabv1
利用双线性插值方法对深度卷积神经网络的粗评分进行上采样，采用全连接CRF算法对分割结果进行优化。
特征提取网络使用的是vgg16，对网络做了一些调整，池化会损失局部区域的信息，因此，仅仅在前三层做步长为2的池化，后面三层池化的步长都只有1，因此，deeplabv1的降采样只有8倍，得到比较粗糙的分数图，然后在这个基础上进行插值，到原图大小，最后使用全连接CRF进行后处理。
空洞卷积
在使用相同的参数量的情况下，获得了更大的感受野！
全连接CRF
在模型的输出后，由于卷积这种操作很难生成清晰的轮廓，卷积是具有平滑性的。所以通过使用CRF对图片进行确定轮廓。
条件随机场：给定随机变量X条件下，随机变量Y的马尔科夫随机场。条件随机场可被看做是最大熵马尔科夫模型在标注问题上的推广。
马尔科夫模型：从一些已知数据或状态中预测最可能的状态或数据。
通俗易懂就是：通过上下文理解这个不认识的东西到底是什么。
CRF的代码属于pydensecrf的库的，可以直接安装。
下面就是deeplabv1的代码
```ruby
class VGG16_LargeFOV(nn.Module):
def __init__(self, num_classes=21, input_size=321, split='train',
init_weights=True):
super(VGG16_LargeFOV, self).__init__()
self.input_size = input_size
self.split = split
self.features = nn.Sequential(
### conv1_1 conv1_2 maxpooling
nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
### conv2_1 conv2_2 maxpooling
nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
### conv3_1 conv3_2 conv3_3 maxpooling
nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
### conv4_1 conv4_2 conv4_3 maxpooling(stride=1)
nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
nn.ReLU(True),
nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
### conv5_1 conv5_2 conv5_3 (dilated convolution dilation=2, padding=2)
### maxpooling(stride=1)
nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
nn.ReLU(True),
nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
nn.ReLU(True),
nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
nn.ReLU(True),
nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
### average poolingnn.AvgPool2d(kernel_size=3, stride=1, padding=1),
### fc6 relu6 drop6
nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
nn.ReLU(True),
nn.Dropout2d(0.5),
### fc7 relu7 drop7 (kernel_size=1, padding=0)
nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
nn.ReLU(True),
nn.Dropout2d(0.5),
### fc8
nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0)
)
def forward(self, x):
output = self.features(x)
if self.split == 'test':
output = nn.functional.interpolate(output, size=(self.input_size,
self.input_size), mode='bilinear', align_corners=True)
return output
```

### segnet
需要关注的只有一个点：在FCN中它提出了使用编码器部分池化对应的索引来完成解码器部分中上采样，避免使用像FCN中那样上采样需要学习，节省了一定的参数量和计算量。
最大池化和降采样降低了特征映射的分辨率。
segnet的解码器使用从相应编码器接收到的最大池化索引来执行其输入特征图的非线性上采样。具体来说：解码器使用在相应编码器的最大池化步骤中计算的池化索引来执行非线性上采样。上采样的特征图是稀疏的，然后与卷积层产生密集的特征图。
！！！这个方案在后续中并没有人使用。那也就没必要看这个代码了。

### DeepLabv2
依然是采用编码器解码器的结构，但是采用了新的编码器。DeepLabv2采用ResNet作为编码器，与之前的版本相比，它具备更深的网络结构和更高的性能。
引入了一个新的ASPP模块，称为空洞空间金字塔池化模块。ASPP模块能够在不同的空洞率下对输入特征进行卷积和池化操作，从而获得更广泛的感受野，然后融合多尺度信息。
模型结构都是和v1一样的~大啦~

### DeepLabv3
全新的ASPP结构，相比与v2，做了更新。我真的？哥们儿就逮着这一个模块薅？
与v2 相比，v3的结构更加复杂，具备更大的参数量。
设计了multi grid方法，其通过在卷积层中使用不同的空洞率来改变过滤器的感受野，从而增加模型对不同尺度特征的识别能力。实现上，multi-grid方法是通过在某些卷积层的空洞卷积中使用具有不同空洞率的卷积核来实现的。
看下面那个代码，你就会发现，这个所谓的multi grid还是用在了ASPP中，也就是self.conv2\3\4这个三个空洞卷积给了不同的具体的空洞率，然后去处理相同的特征图，最后进行不同尺度的特征融合。
哥们儿，真就逮着ASPP可劲儿薅呀？bushi
```ruby
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
        padding=rates[0], dilation=rates[0])
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
        padding=rates[1], dilation=rates[1])
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
        padding=rates[2], dilation=rates[2])
        self.conv5 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        global_features = F.avg_pool2d(x, kernel_size=x.size()[2:])
        global_features = self.conv1(global_features)
        global_features = F.interpolate(global_features, size=conv1.size()[2:],
        mode='bilinear', align_corners=True)
        x = torch.cat([conv1, conv2, conv3, conv4, global_features], dim=1)
        x = self.conv5(x)
        x = self.bn(x)
        x = F.relu(x)
```
### MobileNetV3 & LR-ASPP
首先看下Mobilenetv3与v2的区别
第一方面，就是关于block结构，v3在block结构中加入了一个SE模块。在每个倾斜卷积层之后添加了一个SE模块，用于增强特征表示能力。SE模块就是通道注意力机制。
第二方面也是关于block结构，哥们儿就是在这个通道注意力机制中加入了一个线性瓶颈层，但是看图好像就是一个fc-hard，简单了解下线性瓶颈层。
线性瓶颈结构，就是末层卷积使用线性激活的瓶颈结构（将 ReLU 函数替换为线性函数）。那么为什么要用线性函数替换 ReLU 呢？有人发现，在使用 MobileNetV1时，DW 部分的卷积核容易失效，即卷积核内数值大部分为零。作者认为这是 ReLU 引起的，在变换过程中，需要将低维信息映射到高维空间，再经 ReLU 重新映射回低维空间。若输出的维度相对较高，则变换过程中信息损失较小；若输出的维度相对较低，则变换过程中信息损失很大。
第三方面，调整了last stage。作者删除一个瓶颈层的投影和过滤层，进一步降低计算复杂性。
Lite R-ASPP采用了全局平均池化和SE模块。
### U2NET
这个模型一个开始是用来做突出的物体检测的。但是同样是像素级的分类，所以可以用来做分割。
实际上这个模型就是一个两层嵌套的U型结构，模型的结构看图就能明白。
### transformer
