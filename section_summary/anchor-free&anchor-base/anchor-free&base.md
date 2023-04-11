# 对比anchor-free & anchor-base
起因是这样的：我在完成我的小论文最后模型对的时候，做了一个部分，那就是对于目标检测任务，模型对比，我对比了yolox，但是万万没想到yolox的效果确实真的不错，这让我对anchor-free产生了强烈的好奇。因为很早之前一直都是对于anchor free有一个简单的了解，就是一些表层的了解，甚至于怎么做的都不是很清楚，但是这一次真的让我非常想看一下，它是如何实现的。
所以这篇文章主要就是来看下YOLOv5和YOLOX的网络结构，帮助了解一下anchor-free及anchor-base
## YOLOv5
我们废话不多说，其实可以直接上代码。
```ruby
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]
        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        self.backbone_name  = backbone
        self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)
        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)
    def forward(self, x):
        feat1, feat2, feat3 = self.backbone(x)
        P5          = self.conv_for_feat3(feat3)
        P5_upsample = self.upsample(P5)
        P4          = torch.cat([P5_upsample, feat2], 1)
        P4          = self.conv3_for_upsample1(P4)
        P4          = self.conv_for_feat2(P4)
        P4_upsample = self.upsample(P4)
        P3          = torch.cat([P4_upsample, feat1], 1)
        P3          = self.conv3_for_upsample2(P3)
        P3_downsample = self.down_sample1(P3)
        P4 = torch.cat([P3_downsample, P4], 1)
        P4 = self.conv3_for_downsample1(P4)
        P4_downsample = self.down_sample2(P4)
        P5 = torch.cat([P4_downsample, P5], 1)
        P5 = self.conv3_for_downsample2(P5)
        out2 = self.yolo_head_P3(P3)
        out1 = self.yolo_head_P4(P4)
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2
if __name__ =="__main__":
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    img = torch.rand(1,3,640,640)
    num_cls = 11
    phi = 's'
    model = YoloBody(anchors_mask,num_classes=num_cls,phi=phi)
    out = model(img)
    print(out.shape)
```
我把代码精简了下，留下了主要的部分，首先和anchor-base的最显著的特征就是，我们需要一个anchor_mask，换句话说就是需要一个存放anchor大小的txt文件。其次的不同其实主要在head那部分。
yolov5从整体上看，三部分组成：backbone、neck、head。backbone的输出是：feat1、feat2、feat3。neck的输出是：P3、P4、P5。最后经过yolo_head输出：out1、out2、out3。
这里我们一共要检测的类别数为11，所以最后模型输出的shape为(1,3*(4+1+11),feature map,feature map)。这里为什么要写成(4+1+11)呢，因为可以和后面anchor-free对比上，方便观察。
但是其实anchor-base和anchor-free的最终输出形式上基本没什么差别。
接下来我们看下yolox的anchor-free的网络架构
```ruby
class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width = 1.0, in_channels = [256, 512, 1024], act = "silu", depthwise = False,):
        super().__init__()
        Conv            = DWConv if depthwise else BaseConv
        self.cls_convs  = nn.ModuleList()
        self.reg_convs  = nn.ModuleList()
        self.cls_preds  = nn.ModuleList()
        self.reg_preds  = nn.ModuleList()
        self.obj_preds  = nn.ModuleList()
        self.stems      = nn.ModuleList()
        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels = int(in_channels[i] * width), out_channels = int(256 * width), ksize = 1, stride = 1, act = act))
            self.cls_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = num_classes, kernel_size = 1, stride = 1, padding = 0)
            )
            self.reg_convs.append(nn.Sequential(*[
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act), 
                Conv(in_channels = int(256 * width), out_channels = int(256 * width), ksize = 3, stride = 1, act = act)
            ]))
            self.reg_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 4, kernel_size = 1, stride = 1, padding = 0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channels = int(256 * width), out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
            )

    def forward(self, inputs):
        outputs = []
        pdb.set_trace()
        for k, x in enumerate(inputs):
            x       = self.stems[k](x)
            cls_feat    = self.cls_convs[k](x)
            cls_output  = self.cls_preds[k](cls_feat)
            reg_feat    = self.reg_convs[k](x)
            reg_output  = self.reg_preds[k](reg_feat)
            obj_output  = self.obj_preds[k](reg_feat)

            output      = torch.cat([reg_output, obj_output, cls_output], 1)
            outputs.append(output)
            pdb.set_trace()
        return outputs
```
这里我只是简单的将YOLOX的head列了出来，因为前面的backbone、neck和yolov5的基本没有区别，所以这里就不用再看了，也就是经过neck之后的输出：P3(1,128,80,80)、P4(1,256,40,40)、P5(1,512,20,20)。这些组成一个元组放在了上面代码的inputs中，作为输入。
首先简单看下yolox的初始化。这个初始化，做的很精简，用到了```nn.ModuleList()```也就是我们三个输入需要经过不同的检测head才能输出，但是这里将三个检测头的对应部分存在了对应的```nn.ModuleList()```中。
下面简单列举一下，这里nn.ModuleList()是做什么的。
```ruby
        self.cls_convs  = nn.ModuleList()#这个是为了获取类别信息进行的特征提取卷积
        self.reg_convs  = nn.ModuleList()#这个为了获取位置信息进行的特征提取卷积
        self.cls_preds  = nn.ModuleList()#这个是用来输出类别信息的卷积
        self.reg_preds  = nn.ModuleList()#这个是用来输出位置信息的卷积
        self.obj_preds  = nn.ModuleList()#这个是用来输出置信度的卷积
        self.stems      = nn.ModuleList()#这个是用来调整通道数的卷积
```
其实看过了初始化的过程，相信后面的所有问题都迎刃而解了把。
也就是YOLOX不像yolov5，直接一个卷积把所有的参数都计算出来，而是利用相同的输入特征，进行不同的特征提取，然后分别使用卷积输出三个结果：类别信息、位置信息、置信度信息。就是这么简单
所以后面我们得到的就是(1,16,80,80),(1,16,40,40),(1,16,20,20)。这里的16也就是我们上面说的4+1+11！
所以我们在拿到模型的输出之后，也不需要更改anchor-base模型后面对数据提取的代码啦！

#### 小声评价
我个人任务哈，这里anchor-free能变强，我感觉有很大的可能，就是我们实际上是在检测头上做了更多的操作。因为yolov5的检测头只是一个卷积，而这里的检测头却进行了这么复杂的操作，所以我感觉涨点也是没有问题的，毕竟参数多了很多把。这只是个人见解。还是要感谢yolox，毕竟这个检测头这么写，也是非常的清晰，任务合情合理，还是很棒的。