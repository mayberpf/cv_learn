# DETR
终于！终于对这部分下手啦，我们都知道之前所有的transformer，应用到cv领域一般只是针对上层任务，也就是图像分类，终于！这个DETR就是针对下游的目标检测任务！本篇文章主要分成一下几个部分：1、最近对Attention机制有了新的了解，所以下复习一下。2、简单说一下目标检测任务的相关需求。3、关于detr的论文主要是在说什么事情。4、debug DETR的代码，关注一些新的模块。

## self-Attention机制
首先最近听了一些关于self-attention机制的讲解，还句话说，就是对self-attention这个机制是在图像中是如何起到作用的，进行了解释。下面就是一些我自己的理解后的东西了。
@import "attention.png"
首先如果你对transformer有一点了解的话，你就会知道其中的具备的attention机制，当然在深入一丢丢，你就知道attention机制的计算就是三个变量之间的计算：Q(query)、K(key)、V(value)。当然你会稍微了解到self-attention就是利用这三个变量计算一个值，其计算公式就是图中的上式。接下来我们就针对这个公式，进行解释，想知道为什么这么做，这么做的到底是为了什么。
首先，区别于NLP，CV的输入类似一个长方体，毕竟还有一个通道数，所以为了满足attention计算的要求，我们首先是需要对原始图片的输入进行一些操作（特指Flatten），主要可以起到将二维的图像展开成一维的形状，这样就将原本是长方体的输入形状转变为二维。
当然在做Flatten之前还需要做一个事情，那就是将原始输入放进backbone得到feature map，也就是说后面展开的是feature map，为什么不会针对原始的图像输入进行展开呢，简单说就是考虑到显存的有限，毕竟如果只是一个800* 600的图像，展开可就是48w的数据，到后面qk点积的时候，得到的结果可就是48w * 48w的数据啦，是不是大到离谱了，所以说，attention计算的输入一般都是feature map，或者在计算过程中，会进行下采样(pvt,swin transformer)，类似于直筒结构的vit就会很占用显存。
ok，到这里，attention计算的准备条件基本上就说完了，原始输入经过backbone进行下采样得到feature map，当然，下采样过程中，通道数会增加。，所以feature map在经过展开之后，得到的二维输出的shape是（h * w）* c ，前面也说到了是三个变量，所以展开后的输出需要经过卷积的操作，得到我们想要的三个变量，也就是QKV。这里简单说明就是我认为三个卷积是不同的。
准备阶段进行完之后，就是简单的按部就班的操作，同时也就是我们需要理解一下，为什么按照那个公式去做。
第一件事情，就是我们要知道，qkv这三个输入中的每个值代表什么。qkv他们分别的全称上面已经提到了，想要知道qkv分别代表什么，首先要知道他们是怎么来的，我们假设图片的原始输入是640 * 640 ，那么图片经过backbone下采样也就是提取特征之后，这里我们假设经过backbone后的feature map的大小是20 * 20，也就是经过了32倍下采样。用CNN的话说就是感受野是32*32 ，但是用transformer的话说就是一个patch的大小为32 * 32 。我想这里应该很好理解，之前也说过，在pvt中，使用卷积来切分patch的操作，因此这里的backbone也可以这么理解。经过backbone之后的feature map的结果假设为20 * 20 * 256 ，这是一个立方体，但是经过flatten之后，就会变成400 * 256的二维特征，然后经过卷积，输出qkv，所以说qkv的二维特征中每个点的值实际上可以代表一个patch的特征值。我想如果你理解到这里，那么qkv的公式，就会很快理解了。
上面说到，qkv中，每个点的值可以理解为每个patch的特征值，具体一点，因为qkv是展开后的，也就是说q的shape为400 * 256 ，也就是每一列为一个通道，包含了所有像素值；而每一行表示的相同的一个向素值。那么k的转置不就是反过来，每行是一个通道，包含所有像素，每列表示相同的像素。所以我们在做的q与k的转置的点乘是什么意思呢？ 也就是说，每行乘以每列，不就是像素与像素之间的关系吗！记不记得，这种点积的过程实际上就是相似度的判断，二者相似度越高， 那么结果就越大，反之则越小。所以就很容易理解了，q点乘k就得到了每个像素与其他像素的相似关系，第一行就是第一个像素与其他像素之间的关系，第二行就是第二个像素。到此，也正能说明transformer的全局性。
我们看到哈，qk点乘完之后，得到的feature map的shape是400 * 400 ，它代表的是每个像素与其他像素之间的 关系。然后我们思考为什么要除以一个根号下的维度。这是因为我们在点乘的过程中是第一行乘第一列的对应元素之后将所有加起来，那么如果不除以一个数的话，这个数就会很大，所以这里除以的根号下维度实际上是起到一个减小数值大小的作用，但是为什么是根号下，而不是直接除以一个维度，这个应该是经过模型的实验最后确定的数。
接下来就是softmax，首先softmax应该很熟悉了吧，就是所有数取e为底数的指数计算，然后求所占比值。因此这里最终得到相当于一个归一化的操作，让所有相似程度的值均处在[0,1]之内。
最后一步，就是将我们前面得到的像素与像素之间的相似关系与v相乘。最终得到的不就是一个对应了这个相似关系的新的特征图！具体，怎么对应的，这个可以仔细思考下～类似与上面的q与k的点成。
综上所述！我们这个公式根本就不会对特征图进行降维，也就是我们的输入是多大的特征图，输出就是多大的特征图！只不过这里输出的特征图是有一个全局注意力在里面的！这就是注意力机制！！！
当然，你可能还会在意什么是多头注意力机制呢？我们上面说了一堆废话，实际上就是做了一次qkv之间的计算，我们称这个为单头，那么多头，自然就是将一个输入，进行多个qkv的计算，最后我们将输出结果进行拼接，这个时候shape就又变成长方体了，不过我们可以通过conv进行降维使其变成与输入相同的形状，这就是多头注意力机制。
到此，我们就把注意力机制简单的解释了以下，我感觉把总结一句话就是：不管是单头，还是多头，我始终是不变（爱你）的形状。
## 目标检测
dense prediction 这个词我是在detr中才知道的。用我理解，之前所有的目标检测都是一个套路，不论是一阶段还是二阶段，最终实际上我们得到的结果都是比较密集的！怎么理解这个密集呢， 比如说图片中有一个苹果，但是算法最终会输出很多框，这些框都是表示这个苹果，所以我们需要通过使用nms的操作进行预测框的筛除，最终保留一个最好的框。在detr中作者称这种预测方式为dense prediction。所以说，这篇文章的作者提出了一种真正的end-to-end的目标检测算法，也就是完全不需要nms等操作。这也就是detr这篇论文的中心主旨。
## 论文
@import "detr_paper.png"
这是detr文章中的模型主要结构，实际上和我们在上面提到的注意力机制很像，都是输入经过一个cnn的backbone之后，得到feature map ,然后将feature map和位置信息positional encoding结合起来，我印象中这里的positon 信息，是作者固定死的，也就是学习不到的。具体我们下面可以看下代码。之后又分为encoder和decoder，encoder就是我们在熟悉不过的部分，也就是在vit、pvt中使用的encoder，基本上没有任何改变。所以这里我们主要说一下decoder的部分。
encoder的主要结构
@import "encoder.png"
就是简单的将position encoding加到了QK中，然后进行简单的多头注意力机制和add和归一化，进行FFN等这里的FFN可以简单理解为MLP，就是几个线性层和激活函数。
@import "model.png"
这里的decoder我们可以简单看看它的结构，我们可以倒着看，最后进行的一系列FFN很容易理解，往前就是一个cross-attention，通过图可以简单了解它的QKV是怎么来的。最上面一层就又出现了多头注意力机制。结果这么看下来其实很简单，主要的是看看代码是怎么做的。

说真的如果你能看到这里，那么只可能有两种可能性，第一就是你真的是学习来的，第二我们的关系可能不是一般的好。如果这两种都不是的话，那么我也挺好奇为什么你能看到这里，祝你天天开心吧。因为我准备在这里插入一些与本篇文章要说的东西毫不相关的事情（建议跳过，毕竟不好理解）。其实这篇文章早就在很早之前就应该完成了，但是后面出了一些事。比如说新的一年，我真的立住了一个flag那就是每天早起，所以就是刚到研究院后的一个星期，真的很辛苦，早起晚睡，中午还要刷题。坚持了一个星期吧，但是后面被迫就没有做了。因为基金。但是我也不是很在意做不做免费劳动力，工资多少，毕竟我现在没有工作，对这个并没有很高的要求，但是一开始是有预期的，没有达到预期的话，自然也不会很开心把。对，我有计划，我很庆幸自己知道在各个时间应该干什么，空闲的时候想要做什么。终于也是过了写基金的半个月，还好吧，就是耽误了很多事情，耽误了我喜欢看的代码，耽误了高精地图的项目，其实最主要的是耽误了我的小论文，嘻嘻嘻。iccv嘛没几天了，我猜中不了也没关系，主要我想看看审稿人的意见。哎，人要是能分身就好了。后来为了提高效率，我买了一本去年就想看的书《心流》，好久没看过书了，但是不得不说纸质书的感觉真的不一样，哈哈哈哈。书会慢慢看，但是每章我都会写一些东西，哈哈哈，最后那本书将会成为我的珍藏吧。其实有时候不是我太乐观，也不是我心态好，难听点是我在对自己pua，好听点就是我在安慰自己的心。但是有时候确实很累，我不想说话，我想多听听。所以你会发现，人不快乐很多人时候，是因为在做的事情不能够控制，也不是想做的。《心流》里也说了，（翻了翻书找到了原文）“真正给人带来乐趣的并不是控制本身，而是在艰难状况下行使控制权的感觉。”。所以你知道如何找到这种感觉了吗？毕竟身边总会有一些琐碎的事情，那就劝劝自己，别放在心上。记得用手抚摸自己的心，告诉他一切都会好起来的加油。
鬼知道，怎么回事，我又熬过了一段刻骨铭心的时光，在写过基金的那一段时间，大概十天的时间吧，我知道机会不多，要把握，同样，我也不甘心就这样错过，所以你知道这十天我是怎么过的吗，每天九点之前到研究院，晚上早点就是十一二点，晚点就是快一点，回宿舍。每天重复只做一件事小论文，画图，润色，排版，实验等等。不过还好，提交啦，嘻嘻嘻，其实也不是很难。就想过年的时候说的：关关难过关关过！这是地球，这可不是天堂，我最亲爱的宝贝。我想这周可能还是主要干一些文档的事情把， 计划赶不上变化，即使我在上周六给自己列了近期的计划，看来只能往后推迟咯。

## 代码
首先第一个问题，也是我在跑代码的时候发现的，那就是会提示我torch.nn.Transformer找不到，也就是没有那个包。这个问题我没有详细的去查到底是哪里的问题，但是我发现在我工作的电脑上是可以运行的，但是我自己电脑上就不行，于是我查了一下他们对应的torch版本，结论就是torch1.12可以，torch1.10不行。
我们依然还是分块看，但是今天大概率不能全部搞定了，因为我还没有debug这个代码。我准备看的代码，也就是这个小小的demo，下面这一段很好理解，就是加载模型，加载权重，加载图片，然后检测，看结果。当然，本篇文章也只是针对预测部分进行debug讲解，针对训练部分，后续会再写其他的文章。
```ruby
#调用Toy model
detr = DETRdemo(num_classes=91)
#load model
state_dict = torch.hub.load_state_dict_from_url(
    url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
    map_location='cuda', check_hash=True)
detr.load_state_dict(state_dict)
detr.eval();
#读取图片
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)
#前向
scores, boxes = detect(im, detr, transform)
#查看结果
plot_results(im, scores, boxes)
```
这部分代码容易理解，就是DETRdemo加载的模型。后续我们详细说明，这个是检测的结果。看起来还不错哦。
@import "res.png"
下面就是重头戏了！我们将针对这个demo进行debug，看看这一部分到底是怎么做的。首先我整体看了下代码，我发现这个demo的代码量很少，真神奇。所以我做了一个自认为不错的决定，那就是今天熬夜搞一搞，然后搞一搞训练部分，我就想这部分训练出来能不能放在我的论文里对比一下。那么我们现在开始！！
整体代码可以理解为三部分，第一部分模型，第二部分检测函数、第三部分主函数调用。上面就是主函数，然后调用detect函数，在函数中调用detr模型。搞懂模型当然要知道模型的输入，输入(640,480)的图片。
```ruby
scores, boxes = detect(im, detr, transform)
```
这里调用detect函数，输入为im图片、detr为模型类实例化，transform为张量的转换，主要做的是resize、张量转换、归一化，具体实现结果调用的时候再看，代码如下：
```ruby
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
进入到detect函数，首先就是利用transform将输入数据的格式进行转换
```ruby
    img = transform(im).unsqueeze(0)#PIL格式--->tensor格式（1，3，800，1066）
```
你别说，这里的resize后面只有一个数字，就是对图片的最短边进行缩放，同时不改变长宽比。normalize做的是归一化，其中[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]分别表示的是均值和方差，减去均值，再除以方差，从而加快模型的收敛速度。至于这里的均值和方差是怎么来的，那就是从像imagenet那么大的数据集中提取的。
```ruby
    assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
```
assert 断言函数，相信大家用的都不多，实际上就是做一个判断，也就是它这个模型要求输入的shape不能大于1600，否则就报错啦~
接下来就将输入放到模型里面
```ruby
    outputs = model(img)
```
接下来我们先不着急进入到forward中，先看一下模型的init。
```ruby
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()
        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc
        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
```
这里我们可以将init的一些设置对应到论文中的图中，可以看出作者的backbone使用的resnet50，这里的Attention机制利用的torch.nn里面的库，简单的看一下这个库主要是做什么的，先来看一张最熟悉的一张transformer的流程图
@import "transformer.png"
那么实际上torch.nn集成的transformer只是其中的一部分，就是encoder和decoder那部分，关于position和最后的linear、softmax都不包括。这里我可以理解detr的调用，但是对于detr中的一些部分不太理解。继续看下这个函数的输入和输出吧。首先是输入的维度，其次是多头的头数，然后就是encoder和decoder的个数。
```ruby
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
```
然后初始化的self.linear_class和self.linear_bbox就是论文中流程图中的最后的FFN，self.query_pos应该就是detr流程图中的object queries，至于self.row_embed和self.col_embed应该是position计算用的吧。接下来我们就看forward进行解密。
```ruby
    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        import pdb;pdb.set_trace()
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
```
在forward的前几层的传播，是这样的也就是只针对指定的几层，所以这里我们就不纠结resnet用了啥，如果你想知道其中每一层都有什么，你可以debug一下，然后输出一下每一层和总共有什么，对比一下就ok。到这里，我们看一下输入经过resnet之后的输出shape是torch.Size([1, 2048, 25, 34])
接下来进到
```ruby
        h = self.conv(x)
```
这一个孤单的卷积实际上只是做一个维度的转换，毕竟是1 * 1卷积的。所以这里的输出shape为torch.Size([1, 256, 25, 34]),但是实际上到这里还不行，因为如果想进入到torch.nn.Transformer这个函数中，输入的维度还是不正确的。接下里呢，就是计算position这个事情
```ruby
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
```
初始化生成了两个可学习的参数，根据它的前向传播可以看出来，根据这两个参数生成了position编码。初始化的参数维度为(50,128),最终经过维度的扩展，均转换为(H,W,128),然后在最后一个维度进行拼接，在0、1维度进行展开，扩大一个维度。也就是(25,34,128)--->(25,34,256)--->(850,256)--->(850,1,256)
接下来实际上是transformer的调用，但是我们提前看下self.query_pos的操作。
```ruby
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),self.query_pos.unsqueeze(1)).transpose(0, 1)
```
首先是self.query_pos在init过程中设定为一个(100,256)的可学习参数，思考一个问题，这个参数实际上就是对应论文流程图中的object queries，也就是说在执行nn.transformer时，需要两个输入，一个输入是图像一个输入是object queries，当然还会把position信息加到输入中。所以这里的self.transformer的两个输入就和上面的代码一样，其中object queries，只做了维度的扩展：(100,256)--->(100,1,256)
然而，针对图像的输入，依然是将二维的特征图展开，然后维度变换最终加上position信息。(1,256,25,34)--->(1,256,850)--->(850,1,256)+position
```ruby
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
```
这里我比较好奇一个点，那就是图像输入为什么是那么个维度呢？看了一眼这个博客，参考：（本来想放个链接的，结果不知道是电脑的问题还是啥，链接出不来了）反正就是在博客中说了这么一个事情
@import "nn_trans.png"
注意看里面的两个输入，实际就对应我们的输入，那么注意看下面的注。相信你会更清晰一些，这里的N表示的是batchsize所以是1，那么S代表句子的长度，换到图像领域，不就是特征图展开的长度嘛！同理E就是通道数的维度呀。这样的话，相信你就会更好的理解上面为什么进行那些维度变换了把。
其实这里我并不需要看输出的维度，因为我们在最上面就分析了Attention机制是不会改变输入的维度信息的，所以最终输出的维度，因为是在decoder输出，所以自然维度会和decoder的输入一致也就是(100,1,256)，最终经过transpose的转换，所以最终输出是(1,100,256)
一开始我的理解是100是类别种类数，但是不是这样的类别种类数是一开始设定的num_classes=91，那我就不知道啦，这里通过两个线性层得到类别和box的输出格式如下
cls：(1,100,256)---->(1,100,92)这里是92是因为91个检测物体种类+1个背景
box：(1,100,256)---->(1,100,4)这里的4肯定就是box框的坐标信息咯
这里可能有一个初步的认识，具体怎么提取后面我们看。
```ruby
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
```
这里经过模型输出了outputs，这里的outputs是个字典，里面存放的是cls和box信息，上面这一步是针对cls进行做了一步softmax,然后将最后一个类别去掉了，这里可以将这个类别理解为背景的分类？所以输出维度为(100,91)
然后这一步我好像是懂了
```ruby
    keep = probas.max(-1).values > 0.7
```
<!-- 这么理解，我们现在获取的维度是(100,91)也就是100行91列，然后通过max(-1)获取的是每一个类别的最大特征值，然后将这些值与阈值进行比较，筛选出大于阈值的类别序号。 -->
我真的一整个无语，因为我在debug我发现这输出的keep的shape是(100),也就是这和我尝试理解的不一样，但是却和max(-1)一致，但是我认为这样没有意义呀！
```ruby
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
```
后面这里涉及到一个函数rescale_bboxes，看名字就可以理解， 就是缩放嘛。这里我们先简单跳过缩放的函数把，太晚了
```ruby
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b
```
在缩放之前，做了什么呢，针对上面的筛选结果，对box进行操作。因为我们上面得到了100个True或false，所以这里对应box的第二个维度，最终删选true的box，然后进行缩放即可。
最后的最后就是一个可视化
```ruby
def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()
```
这里就不说了，太晚了。但是我真的不能理解max(-1)那个地方，我目前感觉那么做实际上没有什么意义，只不过是因为那么做了，让模型学到了，才能做目标检测。
那就这样把，这只是针对验证环节，相信在训练环节，会更加复杂，据说会有aux_loss的使用，以及最终的loss是如何计算的，匈牙利匹配是怎么做的，这一切都有待了解！☺