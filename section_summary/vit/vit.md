# VIT
最近终于是抽得几天空闲，能够仔细看看我梦寐以求的transformer，好像自从的这个东西进入到cv的世界的时候，一切的cnn都是那么的渺小，慢慢的好像transformer变得也越来越强大。所以我觉着有必要看看这一部份，以后一定可以大有所为！
本篇的文章呢，我打算从两个方面来介绍transformer：原理论文，代码。当然主要结合代码的介绍，因为在之前的一些文章中，我写过关于transformer的一些文章。
# 原理
我们都知道transformer实际上是起源于nlp的应用，并且transformer在nlp领域好像已经成为了独角兽的存在。所以文章作者想通过transformer的方式来进行cv方向的应用，因此产出了这篇论文：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale。
论文链接：https://arxiv.org/pdf/2010.11929.pdf
但是我感觉，关于这篇论文，真的讲解特别多，大家都会针对它的原理进行讲解，网页上一搜vit随处可见。但是对于代码的讲解却是少之又少。
@import "transformer.png"
这个就是原理图，我们在大佬博客的时候，这种图随处可见。简单来说一下它的原理：对于一张图片而言，先将其分割为N*N个patch，然后将patch进行Flatten，再通过一个全连接层映射成tokens，对每一个tokens加入位置编码（position embedding），会随机初始化一个tokens，concate到通过图像生成的tokens后，再经过transformer的encoder模块，经过多层encoder后，取出最后的tokens（即随机初始化的tokens），再通过全连接层作为分类网络进行分类。
简单说真的很抽象我感觉，接下来我将用自己的语言来描述下vit的原理。首先我们知道vit实际上实现的最基础的图像分类的任务，利用cnn的思想，就是对图片进行特征提取，然后通过一些线性全连接进行分类即可。将这种思想换到transformer，后面的全连接层实际上是没有变的，也就是图中的MLP Head；而特征提取的环节变成了transformer的形式。
接下来我们应该了解一下，nlp中，transformer是怎么做的。举个简单的例子，就是英文翻译成中文，我们猜测，应该是英文进入模型，最终输出中文，没什么问题，但是其实中间还有一步骤，那就是应该将英文转换为一种编码后的形式，然后再利用解码的形式将其转换为中文。有没有感觉这很像分割模型。那么nlp是怎么计算的呢？比如输入一句话：hello world! 模型会将这句话进行拆分，拆分成两个单词，然后对两个单词进行编码，换句话说就是转换成矩阵的 形式，这部分我们没必要了解，因为在图像中和在自然语言中是不一样的。我们需要了解的就是nlp中对每个编码过的单词是怎么进行操作。这里就需要引进qkv了，qkv分别代表：queries查询词、keys关键词、values值。然后通过一个公式计算各个单词的attention。这也是其具有全局视野的原因。
我找了很多博客，信我，这篇图做的很不错：http://t.csdn.cn/HWx32
公式如下：
@import "softmax.png"
通过使用这个公式，计算一句话中每个单词和其他单词之间的关系程度，从而达到attention机制。在cv中也是这样的。那么接下来我们就针对代码进行讲解。我还是建议去debug代码，调试代码是学有效的方法。
# 代码
首先我们需要一份代码：https://github.com/lucidrains/vit-pytorch
这份代码是写的很好的。
vit的代码我们可以在vit_pytorch文件夹下，找到vit.py文件，大体看一下，实际上transformer整体的代码也不过才一百多行，并不是很多。但是我们可能需要加入一个主函数，调用这个函数然后方便我们后续使用和调试。
```ruby
if __name__  =="__main__":
    import torch
    import pdb

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    img = torch.randn(1, 3, 256, 256)
    pdb.set_trace()
    preds = v(img) # (1, 1000)
    print(preds.shape)

```
上面代码中，pdb是用来进行代码调试的库，使用起来还是很方便的，推荐给大家使用。okk，这样我们就可以调用这个函数了，其中，有很多参数，这些参数我们可以先不着急，最后我们再看这些参数的实际含义是啥。
接下来就是进入到vit类的前向传播中，就可以看到模型的整个过程。
```ruby
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        # pdb.set_trace()
        x = self.transformer(x)
        # pdb.set_trace()
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
```
虽然很短，但是里面包含了很多东西，所以接下来我们将进行一行一行的理解。主要是针对前向传播中的操作，首先就是第一行
### 切patch
```ruby
    #初始化
        self.to_patch_embedding = nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        nn.Linear(patch_dim, dim),
        )
        #前向传播
        x = self.to_patch_embedding(img)
```
第一步，我们先简单的概括一下，实际上就是对输入图片进行切patch，然后进行线性转换。我们可以看下主函数，我们的图片输入是(1,3,256,256)，代码中的patch_height,patch_width指的是一个patch的高宽，这是我们一开始可以设置的，就是patch_size,在类初始化中
```ruby
        patch_height, patch_width = pair(patch_size)
```
因此我们可以计算一下256/32=8，因此，我们可以知道，输入的图片经过切patch的操作，可以将其切分成64个patch。关于代码还有一个问题，那就是Rearrange这个函数是啥意思？首先这个函数是在einops库中的，其主要起到的是维度操作的作用，因为平时我们知道针对cnn的一些维度变换，我们多使用的是transpose、permute、reshape等，但是Rearrange实际上有他自己的一个优势，可以去看下它的语法是什么，我个人感觉很好使。简单说下。
```ruby
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
```
就是假设输入维度为(1,3,256,256)这里做切patch，将256分为8 * 32 。也就是代码中的p1 =p2 =  32,h = w = 8,于是输入的维度可以写成(1,3,8 * 32,8 * 32),经过变换，其转换为(1,8 * 8, 32 * 32 * 3)也就是(1,64,3072)，最后我们做fc线性变化得到输出为(1,64,1024)。到这里反映到我们图中，也就是这些部分
@import "1.png"
### 构造patch0
```ruby
        #初始化
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        #前向传播
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
```
在最初的原理图中，我们可以看到作者将图片3*3也就是9个patch，但是最后输入进transformer encoder中却有0到9一共10个部分，也就是其实除了图片切分的部分，还需要一个部分来确定图片类别，所以这部分是cls_token。作者使用```nn.Parameter(torch.randn(1, 1, dim))```的方式构建一个能够学习参数的变量，维度为(1,1,1024)。接下来的repeat实际上就不用多说了，是在batch_size维度上重复，我们的主函数中设定了输入的batch_size为1，所以这里最后输出也就是(1,1,1024)。接下里就是将这一部分和刚才那一部分进行拼接。最后得到维度为(1,65,1024)。
这里就是我们对256的图片进行切分，patch的大小为32，切分后实际上是有64个小块，但是由于需要一个cls_token，所以这里的维度就是(1,65,1024)。到这个地方还不方便看这一部分实际上对应原理图的哪部分。我们在看完后面的pos_embedding再来确定。
### pos_embedding
```ruby
        #初始化
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        #前向传播
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
```
这一部分就是说的位置信息，和上面一样，作者使用相同的方法构造了一个可学习参数，它的维度是(1,65,1024)，后面就是dropout，后面我们可以看到，实际上就是作者是将位置信息通过普通的加法将其添加进去的。
到这里我们就可以看到到这一部分对应原理图中的哪一部分了。
@import "2.png"
如图可知，图片通过patch切分为1到9，然后通过构造patch
0，即可得到10个块，最后每个块再结合位置信息position embedding。
###  Transformer Encoder
将上面三个部分组合后，我们得到的数据维度是(1,65,1024),接下来我们就将进入到encoder中。看图可知，在encoder中，可以主要分为三大类别：attention、mlp、norm。
```ruby
x = self.transformer(x)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
```
整个encoder的代码就是这些。我们可以看到初始化时，初始化了self.layers。这一块中存放了很多层，但是每层都是一样的，是一个循环叠加的问题，循环的次数，就是depth的值，这是我们一开始可以设定的。本文代码中的depth=6。在每一层中，都是固定的。有一个attention和feedforward。在前向传播中，可以看到将每一层的两个部分提取出来，然后进行操作加的操作。接下来我们将针对attention和feedforward两部分进行拆解。
#### PreNorm
```ruby
PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
```
你是不是好奇为什么这里会把Attention这个类放在Norm的类中，我们清楚，norm实际上就是归一化的一个操作，我们先来看PreNorm的代码实现，你就可以解答为什么会有一个类在这里面了。
```ruby
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
```
首先在初始化中，看到，作者进行归一化实际使用的是LN的操作，其次，为什么输入中会有类呢，是因为这块代码在初始化过程中就对attention进行初始化为self.fn。最后再进行调用。也就是先进行归一化再进行attention。
这一块可以理解为：原理图encoder中下图部分
@import "3.png"
同理，我们可以知道每一层除了attention还有mlp，我们知道mlp简单说就是多层感知机，说白了，就是几个全连接层。在这个代码中也是这样的，使用FeedForward类进行mlp的操作，里面其实就是两个线性层。
```ruby
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
```
所以这两部分其实构成了整个encoder的代码。所以就可以很好理解下面这张图片了。
@import "4.png"
#### Attention
接下来我们就对Attention模块详细分析。还记得前面说道的qkv嘛，实际上这个环节就是针对qkv的提取和计算的操作。
```ruby
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
```
这是Attention部分的初始化，接下来我们将对forward进行一步一步解析。
```ruby
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        #得到qkv
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        #切分qkv
```
我们的输入是(1,65,1024)，这个输入经过self.to_qkv，也就是一个线性层，将1024这个维度变成inner_dim*3的维度。这里inner_dim是dim_head与heads的乘积。这里的heads就是多头注意力机制的头数。实际上可以理解为，我们一开始的输入是一个1024的长条，多头注意力实际上将这部分的长条切分成多条进行操作。因此这里将(1,65,1024)经过线性层得到(1,65,3072)然后使用chunk将这个张量在最后一个维度分成3块，所以qkv成为了一个元祖。
```ruby
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        #dot product后除以根号下norm
```
这一部分就是将上面获得的qkv元祖，将qkv分别取出来，并重新进行维度的转换，最后得到的q、k、v的维度均为(1,16,65,64),因为代码中我们初始设定了heads = 16。
```ruby
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
```
这一部分就是对qkv进行计算了，简单说这句话是对Q与K的转置相乘，然后再除以根号下dk。这里就存在另外一个函数einsum。具体使用其实我没有去查，但是可以看出来实际上就是矩阵相乘的意思，``` i d,  j d ->  i j```看最后我猜测实际上会将相同的维度放在中间位置，也就是会对k做转置。然后做矩阵相乘。
```ruby
        #softmax
        attn = self.attend(dots)
```
这一步分好理解就是softmax
```ruby
        #乘到v上，再sum，得到z
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        #整理shape输出
```
这一步就是最后的计算过程，也就是乘V。这样计算完之后我们得到的最后的维度为(1,16,65,64)
```ruby
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
```
最终将这部分的维度转会一开始的构造，也就是(1,65,1024)。这里有一个self.to_out其实应该是做的维度的改进统一，但是我真的不清楚为什么要做这个。可能是在qkv计算之后，维度会变。
到这里Attention就进行完了，然后进行操作加。

#### feedforward
这一部分实际上就是简单的两个fc层和一层激活函数。
```ruby
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
```
到此为止，我想整个vit就要接近尾声了。因为在进行完了encoder之后，就是简单的mlp head进行分类了。
### MLP Head
```ruby
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
```
最后我们对输出进行分类操作，代码中提供了两种方式，通过self.pool进行操作，可以选择mean和cls。其中cls实际上就是取出我们构造的patch0，然后输入mlp_head进行类别分类。这里通过cls_token进行分类我理解，但是使用mean好像有点像cnn的操作了哈。
这里的```self.to_latent(x)```不太明白是干啥的，但是它似乎并没有对输入进行什么操作。```self.mlp_head(x)```这里的mlp分类器就是单纯的LN归一化+全连接层的使用。
## 总结
我想到这里就结束了，实际上在代码中还有一个利用transformer进行猫狗分类的操作，有兴趣可以试一试。相信我，现在已经没有人比我更懂transformer。
当然未来我还会继续学习transformer相关的东西，主要可能包括：PVT、swin transforme、Detr、Deformable Detr、spare等等。

