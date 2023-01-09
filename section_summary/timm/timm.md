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