# Timm
仔细看看，这是timm不是timi！！！好啦，言归正传，相信如果你是在做深度学习，那么timm这个库你多少应该会听说过，或者见过。况且，这个库在kaggle中出现的频率更高！！！几乎牛逼一点的人，都会直接调用这个库，针对模型进行调用和简单的更改。一开始就说这些感觉很难理解，接下来我们可以直接看一点点的代码，来了解这个库可以如何使用。
这块代码简单说，就是针对efficientnet的一个调用。同时因为分类类别的问题，num_class==1，因此模型的最后的分类器需要更改。其实不瞒大家，那就是这是我最近在RSNA比赛中找到的一个baseline。里面注释掉的代码是我后来添加的，因为其实我本身对于timm这个库的使用也不是很懂，所以才写了这个文章了解一下，其实写这个文章的初始动力是：我想知道在kaggle的平台怎么直接使用timm。
```ruby
class RSNAModel(nn.Module):
    def __init__(self, model_name):
        super(RSNAModel, self).__init__()
        # self.efficient_b1 = create_model(num_classes=1).to(device)
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if "efficientnet" in CFG.model_name:
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, CFG.n_classes)
        elif "resnet" in CFG.model_name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, CFG.n_classes)
    def forward(self, img):
        return self.model(img)
        # return self.efficient_b1(img)
```
简单解释一下，首先我知道timm这个库是通过使用```create_model```创建模型的，也就是在timm中包含了很多模型，我们需要调用模型的话，只需要将模型的名字写对就可以了。然后这里看代码就可以知道，实际上作者提供了两个模型的调用，调用哪个取决于```CFG.model_name```同时和上面说的一样，这个比赛实际上做的就是一个图像分类的问题，只有两个类别：有癌症、没有癌症。因此这里需要更改调用模型的分类器，这里说到的分类器我们后面再看是什么。首先获取模型分类器的输入```in_features = self.model.classifier.in_features```,其次将模型分类器改成了一个linear层```self.model.classifier = nn.Linear(in_features, CFG.n_classes)```就这样，我们就将整个模型调用进来啦。最后就是进行前向传播就好啦。
其实看到这里，相信大家对于timm这个库已经有了初步的认识。接下来我们可能要详细说一说这个库啦。
这里关于timm这个库的安装，没必要说了，pip就行。
还是那句话，真的推荐大家去看官方的文档：https://timm.fast.ai/
毕竟，这个笔记就是我在看官方教程记得笔记。接下来我们就按照文档的步骤进行学习。
```ruby
model = timm.create_model('resnet34')
```
创建模型，刚才已经说过了，如果需要加载预训练模型的话
```ruby
model = timm.create_model('resnet34',pretrained = True)
```
加载预训练模型的话，代码第一次运行时，就会下载相应的权重。下载地址会显示在终端的log中。
```ruby
model = timm.create_model('resnet34',pretrained = True,num_class = 10)
```
在加载模型时，可以直接选在分类类别数。
```ruby
timm.list_models()
#也可以使用通配符进行搜索
timm.list_models('*resnet*')
```
如果你不知道模型调用的时候名字都叫什么，毕竟有些模型后面还有一些数字，比如efficientnet_b3、resnet18等，可以尝试搜索一下，你会发现光resnet就有一堆。
写到这里发现，其实还有一个库，也是很常用的:fastai，之后我们再说把。其实你看官方文档，你会发现timm其实不仅仅是调用模型还可以设置Data、Loss、Optimizers、Schedulers等。但是在实际的使用上，我们可能大多数都是使用timm进行模型的调用，毕竟比较方便，而且包含的模型是真的多。今天有点事，只能写到这里了。
ok,昨天没干完的事情今天干！！！
在官方文档中，解释了如果使用timm进行自己数据的模型训练，接下来我们简单做个总结：
首先我们需要清楚一个问题，那就是timm在创建模型时，会有很多参数，这些参数存放在一个叫Namespace中，我们可以在其中添加参数，以满足我们对训练策略的修改。
1、如果你需要分布式训练，也就是对应单机多卡```arg.distributed True```
2、设置seed获得可重现的结果
3、使用```timm.create_model```创建模型
4、根据模型的默认配置设置数据配置。这里模型的默认配置一般是这样的
```ruby
{'url': '', 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7), 'crop_pct': 0.875, 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'first_conv': 'conv1', 'classifier': 'fc'}
```
5、设置增强批量拆分----这里我不是很理解，建议了解一下辅助批量规划。
6、如果对多个GPU进行训练，设置apex syncBN或PyTorch原生SyncBatchNorm来设置同步归一化。这意味这我们不是在每个单独的GPU上对数据进行归一化，而是跨GPU的一个范围对整个批次进行归一化。
7、如果需要，使模型可导出torch.jit
8、初始化优化器
9、设置混合精度
10、继续训练的设置
11、设置模型权重的指数移动平均值
12、设置分布式训练
13、设置学习率调度器
14、创建训练和验证的数据集
15、设置Mixup/Cutmix数据增强
16、创建训练数据加载器和验证数据加载器
17、设置损失函数
18、设置模型检查点和评估函数
19、训练和验证模型，并将评估指标存储到输出文件中
上面和模型训练基本没什么差别。我感觉看看有个了解就行了，后面我们针对models进行学习
## Model
这一部分在上面我们已经初步了解了，接下来我们进行深入分析。我们知道可以使用torchvision进行模型的调用，但是有一个致命的问题，那就是调用的模型的是写好的，一些输入的接口要求图像是单通道，那输入就必须是单通道，但是图像一般不都是RGB三通道的嘛，所以问题就来了。同样的道理，如果我们想要让模型的输入是多个通道的，比如30通道，那么基本上就很少有模型进行适配了吧。
这个时候就出现了timm！！它有方法去解决这个问题。
```ruby
>>> m = timm.create_model('resnet34',pretrained = False)
>>> m
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#================改变输入通道数==============#
>>> m = timm.create_model('resnet34',pretrained = False,in_chans = 1)
>>> m
ResNet(
  (conv1): Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```
从上面代码中可以看到，我们在设置in_chans = 1，模型的第一层卷积的输入通道数就会发生改变。所以就是酱紫简单。
但是接下来我们会面临第二个问题，那就是预训练权重的加载问题，因为我们更改了模型的一部分。
timm提供了两个方法去做这个事情。首先timm会判断输入通道数是不是3，不是3的话，就会有两个方法，一个方法就是如果通道数是1，那么就会将预训练权重的3个通道做一个求和，这样就转变一个通道。第二个方法，就是当输入通道数为8或者更大时，会将预训练权重进行复制，然后取出来前8个来做预训练权重。
接下来我们学习非常重要的一个部分，那就是将各个模型变为特征提取器。我们都知道在深度学习中，网络模型可以简单分为三部分：backbone、neck、head。其中backbone重要起到的作用就是特征的提取，所以我们可能需要将不同模型进行替换，看谁的特征提取能力更适合我们的任务。所以接下来就针对这一块进行实现。
也就是我们可以针对模型的具体的层数进行特征的提取。
```ruby
import timm 
import torch 
x = torch.randn(1,3,224,224)
feature_extractor = timm.create_model('resnet34', features_only=True, out_indices=[0,1,2,3,4])
out = feature_extractor(x)
```
这里在创建模型时添加了参数```features_only=True, out_indices=[2,3,4]```，这里我们就是通过设置feature_only来调用返回网络的中间特征图的。然后通过设置索引来确定最后返回的哪些层的feature_map。在上面代码中，可以看到返回的是第2、3、4层。然后我们如何确定这里每层对应的位置呢？我个人理解是：每进行一次下采样就是一层。下面就是上面代码对特征进行提取之后，获取的每层的feature map的大小。
```ruby
>>> out[0].shape
torch.Size([1, 64, 112, 112])
>>> out[1].shape
torch.Size([1, 64, 56, 56])
>>> out[2].shape
torch.Size([1, 128, 28, 28])
>>> out[3].shape
torch.Size([1, 256, 14, 14])
>>> out[4].shape
torch.Size([1, 512, 7, 7])
```
接下来我们简单看下timm在进行create_model时的内部是如何运行的！
其实可以去看下timm原码的GitHub：https://github.com/rwightman/pytorch-image-models
也就是说，实际上代码中内部有一个字典，在我们使用create_model函数之后，根据我们输入的名字进行函数的调用。然后在代码中，各个函数实际上是存放在model中，所以实际上我们可以通过两种方法进行模型的调用。
```ruby
import timm
from timm.models.resnet import resnet34
# using `create_model`
m = timm.create_model('resnet34')
# directly calling the constructor fn
m = resnet34()
```
那么在源码中，resnet34的调用是这样的
```ruby
@register_model
def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    """
    model_args = dict(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet34', pretrained, **model_args)
```
看！这里出现了装饰器，这个装饰器是做什么的呢？因为_model_entrypoints一开始是一个空字典。装饰器register_model做的是将模型的名字添加到_model_entrypoints中。这里我尝试去打印_model_entrypoints这个字典看一下，但是不行！！我不理解。
然后我们通过resnet34这个函数，设定参数最后调用_create_resnet函数
```ruby
def _create_resnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        ResNet, variant, default_cfg=default_cfgs[variant], pretrained=pretrained, **kwargs)
```
在这个函数，又调用了一个build_model_with_cfg的函数，根据字面意思可以知道，就是搭建模型根据我们的参数，这里的参数包括模型函数类，模型名称、模型的默认参数等。
```ruby
def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        default_cfg: dict,
        model_cfg: dict = None,
        feature_cfg: dict = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Callable = None,
        pretrained_custom_load: bool = False,
        **kwargs):
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    model = model_cls(**kwargs) if model_cfg is None else model_cls(cfg=model_cfg, **kwargs)
    model.default_cfg = deepcopy(default_cfg)

    if pruned:
        model = adapt_model_from_file(model, variant)

    # for classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        if pretrained_custom_load:
            load_custom_pretrained(model)
        else:
            load_pretrained(
                model,
                num_classes=num_classes_pretrained, in_chans=kwargs.get('in_chans', 3),
                filter_fn=pretrained_filter_fn, strict=pretrained_strict)

    if features:
        feature_cls = FeatureListNet
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.default_cfg = default_cfg_for_features(default_cfg)  # add back default_cfg

    return model
```
这里就是build_model_with_cfg函数。ok结束。