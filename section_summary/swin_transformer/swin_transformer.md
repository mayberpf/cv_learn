# Swin_transformer
最近真的有努力在看transformer相关的模型，因为从很早之前就说学一学这个东西，但是无奈学不明白，看不懂。所以这次决定认真的来学一学，之前也是看过了vit以及它的代码。目前有了大部分的了解。本篇文章呢，主要针对vit之后改进的一个总结，概括来说，就是我会简单的说说pvt，然后详细说说swin_transformer。简单说说就是说一下原理，详细说说就是原理＋代码，简单明了。
# 直筒型和金字塔型
在解释针对vit的改进之前，我们首先要了解的就是为什么vit需要改进，它哪里做的不好。从标题就可以很快速的概括出，vit的问题所在。我们都知道普遍的CNN模型，我们的输入会随着网络变深，featuremap逐渐减小，也就是一开始我们输入的224 * 244，但是经过网络最后一层featuremap可能就是8 * 8。画出来就像一个金字塔一样，因此我们称这种结构为金字塔结构。同时正是由于这种金字塔结构，使得cnn模型能够更容易使用一些适用金字塔的neck结构，比如我们熟知的FPN。但是反观vit，它的问题就在于它输入图片后，其将图片切分patch，然后每个patch有对应的qkv进行计算，直到最后的计算结束，patch总数都不变，因此我们称其为直筒型。
从上面简单的介绍，就已经暴露出直筒型的缺点：1、计算量大，显存占用明显，因为feature map不会变小，自始至终都是一样大的。2、直筒型并不容易使用FPN等结构。毕竟像FPN这种能够提升很多点的结构，不论是针对CNN还是transformer都是很重要的。因此主要针对上面这两个问题，大家展开了深入的讨论，并诞生了pvt和swin_transformer。
@import "1.jpg"
# PVT
我们可以简单回忆一下vit的结构，将图片切分，然后linear产生qkv，计算qkv，最后进入mlp，循环操作，最终拿出cls_token进行分类任务。实际上很好想到，那就是这一部分最大的计算量就在于qkv计算那个部分，我们假设有n个q，n个k，那么在计算的时候这里计算量就是n方级别的啦。因此为了更好的减少计算量，PVT的作者想了一个方法：限制k和v的个数。没错，那这一部分在代码中的实现，我们就不对着原代码一点一点看了，简单说下：vit在构建qkv时，是通过使用切patch，确定patch数量之后确定qkv对应的数量的，但是在PVT中，作者为了限制kv的数量，在第一次切patch完成之后，再一次切patch来确定kv和数量。就是这么的简单，放到代码中不过也就是一两行的事情。
除此之外，PVT的作者另外一个创新就是：实现了金字塔的结构。这一部分作者是通过：反复切分patch实现的。简单说就是第一次切分完patch放进transformer block中进行计算，作者将这个过程叫做一个stage，PVT中有4个stage，也就是需要切分4次，后面每一次切分都是在前面切分过的基础上进行切分，因此feature map实际上是越来越小的。
@import "PVT.jpg"
实际上我认为PVT通过多次切分实现金字塔结构是完全可以接受的，而且我觉着还是一个不错的方法，但是无奈就是参数量过大，作者采取了限制kv个数的方法，这个方法感觉有点无奈之举。相比于swin_transformer的方法，我们就会发现，实际上这部分是很难权衡的，因为如果你想要获取全局信息，但是又想参数量小，有点鱼和熊掌不可兼得的道理。接下来我们就详细介绍一下swin_transformer。
# Swin_transformer
相比于PVT，swin_transformer并没有限制kv的个数，我个人感觉它限制了感受野。我们前面也分析了，transformer的计算量主要来自于qkv的计算，因此想办法减少qkv的个数尤为重要。PVT实际上就是简单的通过线性层限制了kv的个数，但是swin transformer则是通过限制感受野的方式来限制qkv的个数，换句话说：VIT、PVT都是对整个输入图片进行patch切分，然后计算每个patch的qkv之间的关系。但是swin在整张图片的基础上切分了Windows，也就是在计算过程中，只需要计算各个窗口下的patch之间的qkv之间的关系。
@import "swin.jpg"
就像图中左侧看到的一样，红色框表示窗口，灰色框表示切分的patch，也就是在计算qkv时，不同窗口包括的patch之间并不进行qkv的计算，这样就会大大减少的计算量。但是你可能会好奇，transformer自身的优势就是能够获取全局信息，那这样提取窗口进行计算，就会让其失去全局的感受野，这样岂不是失去了本来的优势。因此作者考虑到这一点，增加了第二部分，那就是窗口的平移的操作，也就是上图中第二部分，将窗口进行平移，得到与之前Windows不同的操作，这样就可以实现跨窗口的计算，也就实现了全局感受野的计算。但是我个人感觉即使这样做，也很难实现像vit、pvt那样的全局感受野吧。
@import "swin_1.jpg"
上面这个图实际上就是实现swin_transformer的网络架构，可以看出其仍然是4个stage，每个stage再次进行patch的切分，以此来实现金字塔结构。swin的创新主要在于transformer block中，也就是上图右侧的W-MSA和SW-MSA。这两块分别实现的就是不平移窗口的qkv计算和平移窗口的qkv计算。模型其他部分看似没有什么特别的啦，但是看代码的话，会发现一些其他的。比如：作者代码中是如何区分shift为不为0的qkv计算、以及一部分关于patch_merging和相对位置的一些操作。因此接下来我们就详细的看看代码。
# Swin_transformer模型代码详解
终于！！！我用了很长的时间，大概两天的时间，终于对swin_transformer的代码有了一个比较清晰的认识，如果下面有什么地方说的不是很对，欢迎大家评论。看了swin的代码，给我感觉就是这个代码真的不是那么简单，有很多地方我看了很久都没有看懂，然后是去网上找一些大佬的文章，才能对一些地方有稍微一点的了解，就比如相对位置那个地方，真的很难。不过没有关系，现在我基本上掌握了感觉。接下来我们会从整体的框架到细节模块进行分析。
说到这里我们就不得不上一个swin的网络结构图，就是上面的那张图，我们可以看到整体的框架由以下几部分组成：切patch、线性embedding、swin transformer block、patch merging组成，其中，重点在swin transformer block中，包括了我们常见的LN、MLP还有创新的W-MSA和SW-MSA。如果你对于vit有一个初步的了解，相信看了这个图大体能明白作者对哪里做了优化改进，如果没有了解，我建议先去看看vit的文章，因为这里为了减小篇幅，我们会省去很多vit中说过的事情。
我们拿到一个模型代码，肯定会找输入，然后一点点往里走，根据forward。接下来我们的代码分析也是这样进行的。为了更好的调试代码，我写了一个主函数，来调用模型。
```ruby
if __name__ =='__main__':
    model = swin_base_patch4_window12_384()
    # pdb.set_trace()
    img = torch.randn(1,3,384,384)
    pred = model(img)

@register_model
def swin_base_patch4_window12_384(pretrained=False, **kwargs):
    """ Swin-B @ 384x384, pretrained ImageNet-22k, fine tune 1k
    """
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), **kwargs)
    return _create_swin_transformer('swin_base_patch4_window12_384', pretrained=pretrained, **model_kwargs)

def _create_swin_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        SwinTransformer, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model
```
这里有一个不太熟悉的东西，那就是装饰器，为了能够更好的了解这个东西，我简单写了一个代码去帮助理解，实际上这个就是python代码中会用到的一个东西，也可以去搜一下，很多文章讲解。
```ruby
def func2(func1):
    def wrapper():
        print("进入func2函数")
        func1()
        print("done")
    return wrapper

@func2
def func1():
    print("进入func1函数")

func1()
```
简单解释一下，就是我们可以通过直接调用func1这个函数来完成func2函数的功能。因此上面，我们通过调用swin_base_patch4_window12_384，就可以调用整个模型，实际上呢，就是在模型调用的基础上，加上了模型参数的定义，包括一些patch_size=4, window_size=12, embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32)等等。这写都好说，接下来我们就直接进入到forward。在调用模型的时，用到了build_model_with_cfg，这是一个timm库中的函数，不得不说timm这个库是真的有用呀，很多东西都可以调用它，有机会一定详细看看它都包括什么。这里简单查了一下，这个函数就是方便我们调用模型的一个函数，同时可以写入参数。这里我们可以就简单当做一个调用模型的函数。因此，我们进到SwinTransformer这个类中，去看它的forward，你会惊奇的发现，十分简单。
```ruby
    def forward(self, x):
        # pdb.set_trace()
        x = self.forward_features(x)
        x = self.head(x)
        return x
```
这里的前向传播就是简单的分成了两个部分，首先第二部分，超级简单就是一个linear做分类器
```ruby
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
```
因此代码中重点的部分主要就集中在forward_features中，下面就是其前向传播的过程。
```ruby
    def forward_features(self, x):
        x = self.patch_embed(x)#1,3,384,384--->1,9216,128
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            # pdb.set_trace()
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x
```
可以剧透一下，就是这里面的重点就是self.layers那个循环那里，其他的地方都比较简单，当然transformer block也是在这里layers里面。我们先跳过那一块看别的。首先是self.patch_embed()这个的作用就是实现了切patch的功能。这里说到切分patch，可以简单说一下PVT和swin对于切分patch是使用的conv的方法，和vit是不同的。
```ruby
self.patch_embed = PatchEmbed(
    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
    norm_layer=norm_layer if self.patch_norm else None)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x
```
看上面代码就很清楚，调用了一个conv进行切patch，切分完之后展开改变维度。我们就用384输入的那个进行举例子。这里卷积用的是4 * 4卷积，也就意味着每个patch的大小是4 * 4。因此我们卷积后得到的就是96 * 96 个patch。
接下来我们继续前向传播，接下来是self.ape的判断，然后加上绝对位置信息，作者在论文中说这一部分会影响性能，所以这块是False。我们也就不用看了。
接下来的self.pos_drop实际上就是简单的正则化dropout
```ruby
self.pos_drop = nn.Dropout(p=drop_rate)
```
接下来self.layers我们先跳过，继续看,后面这里一个是LN，一个是平均池化，然后展开。最后就return了。return之后呢，就是我们一开始说的head进行linear线性分类，输出结果。
```ruby
        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
```
整体的流程大体就是这样的，但是如果只是了解到这里，那我只能说transformer基本就是没看！因为最关键的东西都在self.layers中。因此我们可以打印一下self.layers看看这里面都是什么。这里的self.layers实际上是有4层，这里我们就只显示一层，因为你会发现基本上层与层之间很相似。
```ruby
ModuleList(
  (0): BasicLayer(
    dim=128, input_resolution=(96, 96), depth=2
    (blocks): ModuleList(
      (0): SwinTransformerBlock(
        dim=128, input_resolution=(96, 96), num_heads=4, window_size=12, shift_size=0, mlp_ratio=4.0
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (attn): WindowAttention(
          dim=128, window_size=(12, 12), num_heads=4
          (qkv): Linear(in_features=128, out_features=384, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=128, out_features=128, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
          (softmax): Softmax(dim=-1)
        )
        (drop_path): Identity()
        (norm2): LayerNorm((128,), eps=1e-05, SwinTransformerBlock
          (fc1): Linear(in_features=128, out_features=512, bias=True)
          (act): GELU(approximate=none)
          (fc2): Linear(in_features=512, out_features=128, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
      (1): SwinTransformerBlock(
        dim=128, input_resolution=(96, 96), num_heads=4, window_size=12, shift_size=6, mlp_ratio=4.0
        (norm1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (attn): WindowAttention(
          dim=128, window_size=(12, 12), num_heads=4
          (qkv): Linear(in_features=128, out_features=384, bias=True)
          (attn_drop): Dropout(p=0.0, inplace=False)
          (proj): Linear(in_features=128, out_features=128, bias=True)
          (proj_drop): Dropout(p=0.0, inplace=False)
          (softmax): Softmax(dim=-1)
        )
        (drop_path): DropPath()
        (norm2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (mlp): Mlp(
          (fc1): Linear(in_features=128, out_features=512, bias=True)
          (act): GELU(approximate=none)
          (fc2): Linear(in_features=512, out_features=128, bias=True)
          (drop): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (downsample): PatchMerging(
      input_resolution=(96, 96), dim=128
      (reduction): Linear(in_features=512, out_features=256, bias=False)
      (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    )
  )
```
首先我们说明一个问题，那就是为什么是4层，而且为什么每层有多少个SwinTransformerBlock是谁控制的，看似深奥的问题，实际上就是一个元组的问题，还记得一开始我们调用模型给到的参数depths=(2, 2, 18, 2)。这就很简单，一共四层，每层分别有2、2、18、2个transformer block模块。这里又需要解释的就是为什么都是偶数个。这就涉及到swin_transformer的原理了。简单说就是transformer block根据W-MSA和SW-MSA分为了两种，也就是必须两种block都存在才能更好的实现作者想要的效果。因此这里是偶数，那么我们也就不用在解释，每一层中的block都是谁了。每一层实际上都是先进行W-MSA的block然后进行SW-MSA的block，做循环。接下来我们就会详细介绍一下W-MSA和SW-MSA。
首先做的是取出self.layers里面的每一层，这里我们就拿第一层举例子，因为后面三层和第一层的差别基本没有。
首先我们再看一下self.layers是怎么init的
```ruby
self.layers = nn.ModuleList()
for i_layer in range(self.num_layers):
    layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                        input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                            patches_resolution[1] // (2 ** i_layer)),
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=self.mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                        norm_layer=norm_layer,
                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                        use_checkpoint=use_checkpoint)
    self.layers.append(layer)
```
很明显，这里面调用了一个BasicLayer类，我们进入到这个类里面看他的forward。
```ruby
def forward(self, x):
    for blk in self.blocks:
        if self.use_checkpoint:
            x = checkpoint.checkpoint(blk, x)
        else:
            x = blk(x)
    if self.downsample is not None:
        x = self.downsample(x)
    return x
```
这里面主要有两个部分，一个是self.blocks和self.downsample，上面的use_checkpoint我们这里就忽略了，我没注意它是什么，但是在代码运行的时候，就没进去过。我们直接进入难点，很明显transformer的block都是在blk中进行的。因此我们看self.blocks的init
```ruby
self.blocks = nn.ModuleList([
    SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                            num_heads=num_heads, window_size=window_size,
                            shift_size=0 if (i % 2 == 0) else window_size // 2,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop, attn_drop=attn_drop,
                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                            norm_layer=norm_layer)
    for i in range(depth)])
```
看的出来，这里的self.blocks实际上调用了SwinTransformerBlock这个类，在init代码的最后有一个range(depth)我们就可以知道self.blocks中有几个SwinTransformerBlock完全取决于depth元组。depth=(2,2,18,2)，所以这第一个self.blocks中是有两个SwinTransformerBlock模块的。这里我们再注意一个问题，那就是shift_size也是根据循环设定的。这里我们简单说一下，就是如果shift_size==0，那么transformer block就运行W-MSA那个模块，如果shift_size!=0那么transformer block就运行SW-MSA那个模块。这里看代码我们可以看出，在循环的过程中，shift_size是第奇数个为0，第偶数个不为0。也就是说第奇数个是W-MSA那个block，第偶数个是SW-MSA那个block。
接下来我们进到SwinTransformerBlock中，看其前向传播是怎么做的。
```ruby
def forward(self, x):
    H, W = self.input_resolution
    B, L, C = x.shape
    assert L == H * W, "input feature has wrong size"
    shortcut = x
    x = self.norm1(x)
    x = x.view(B, H, W, C)
    # cyclic shift
    if self.shift_size > 0:
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
    else:
        shifted_x = x
    # partition windows
    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
    # W-MSA/SW-MSA
    # pdb.set_trace()
    attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)
    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```
我们还是优先说简单的容易理解的，前向传播首先整了一个shortcut，看到这个词，就应该明白了，那就是残差，我们可以对照上面的网络架构图，这个shortcut实际上完成的是LN---W-MSA那一块的残差。看一下前向传播的最后
```ruby
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
```
这里的self.drop_path()实现的就是DropPath的操作，简单说可以理解为正则化dropout，只不过实现的方法不一样。接下来最后一行就是进行LN---MLP然后残差。最后return出去，也就是这一个前向传播可以完成一个transformer block。接下来我们将详细说一下这里的transformer block，首先我们在进行第一个block时，shift_size是0这点我们需要清楚，然后我们可以一点一点看了。首先将输入进行LN然后reshape到(1,96,96,128),由于这里shift_size等于0，所以不会进行torch.roll这个操作，只是把x原封不动搬下来了，接下来，进行window_partition。
```ruby
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)#contiguous()深拷贝
    return windows
```
简单来说，这个函数只是实现一个window的切分，当然window_size也是我们一开始就设定好的，这里认为window_size为12，最终，这一个函数就是将输入reshape的一个过程：(1,96,96,128)--->(1,8,12,8,12,128)--->(1,8,8,12,12,128)--->(64,12,12,128)最后我们看到64对应window数量，12,12对应的是一个window的token数，这里通道数128在后面大家就知道会成为什么啦。切分完成之后，最后在reshape到(64,144,128)。然后就是最重要的啦，那就是self.attn。看名字我们就能知道这实际上就是Attention计算环节，也就是我们都知道的qkv的计算。
```ruby
#self.attn的init
self.attn = WindowAttention(
    dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
```
```ruby
#WindowAttention类的前向传播
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        #print(x.shape)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        # pdb.set_trace()
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```
我们的输入是(64,144,128)这里的通道数128，就是在这里经过linear增加到384，然后重新reshape到3 * num_heads * C//num_heads ，这里就得到了传说中的qkv。通过代码可以看到
```ruby
self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
```
将通道数增加到了3倍。也就是变量维度的变换是：(64,144,128)--->(64,144,384)--->(64,144,3,4,32)--->(3,64,4,144,32)换句话说，这里得到的64个window下的144token，每个token的qkv长度都是32。后面其实很简单就是那个我们熟悉的公式，完成了qkT * dk**0.5。这里得到的attn 的shape为(64,4,144,144)。接下来就是一个超级超级难的地方，那就是相对位置编码的加入。这一块，坦白来说我自己看代码基本就是看不懂，最后只是看了一个大佬的文章，才勉强懂了那么一点。
先上链接：https://zhuanlan.zhihu.com/p/577855860
然后我们先简单说一下，这里的相对位置是怎么操作的。如果你不是很想知道这部分具体是怎么做的，看这一部分就足够了。首先有两个参数，一个相对位置索引的参数，这个是不可以学习的；一个是相对位置偏执的参数，这个是可以学习的。那么这里主要是实现一个事情：根据每个token相对于其他token的相对位置索引，来确定相对位置偏执是多少，也就是说两个token相对于某一token的相对位置一致，那么它们的相对位置偏执也是一样的。attn的shape为(64,4,144,144)因此最终取出的map大小也是(4,144,144),最后增加一个维度，然后相加即可。
我知道大概率是看不懂。你只要知道加进特征的偏执是可以学习，同时其根据不可学习的相对位置索引确定哪个偏执。
接下来我们详细说说这一部分相对位置怎么实现的。首先当然看一下这一部分init的代码。
```ruby
# define a parameter table of relative position bias
self.relative_position_bias_table = nn.Parameter(
    torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

# get pair-wise relative position index for each token inside the window
coords_h = torch.arange(self.window_size[0])
coords_w = torch.arange(self.window_size[1])
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
relative_coords[:, :, 1] += self.window_size[1] - 1
relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
self.register_buffer("relative_position_index", relative_position_index)
```
代码整体可以用四个字来概括：晦涩难懂。首先我们要清楚一个问题，那就是相对位置在二维图中是怎么计算的。首先我们要知道绝对位置是怎么确定。比如一个2 * 2的图片，我们将左上角的位置记为(0,0)，那么右下角的位置记为(1,1)这个就是绝对位置。那么相对位置就是做一个减法。比如右下角相对左上角的位置，我们就用左上角的绝对位置减去右下角的绝对位置，最终结果为(-1,-1)。
后面是我自己的理解，仅供参考。关于相对位置，我是这么理解的，因为我们在计算中，只是关心一个window中的各个patch之间的qkv之间的关系，因此实际上这里的相对位置关系，实际上就是一个window下，每个token和其他token的相对位置关系，也就是window_size为12，那么就会有144个token，那么这个位置关系的feature map就会是144 * 144。
因此在实际上计算时，也就是init时，相对位置的index是如何产生的成为了一个难点。根据理解，首先我们需要一个window_size大小的横纵坐标，也就是coords_h和coords_w。接下来我们通过meshgrid和矩阵拼接生成一个网格，这个网格包含的就是横纵坐标。这里的shape为(2,12,12)其中2表示2横纵坐标，12,12表示横纵坐标都是12 * 12 的。接下来我们将其展开--->(2,144)，接下来就是利用广播机制，实现相对位置的计算，也就是上面说的减法。coords_flatten[:, :, None] - coords_flatten[:, None, :]这里在不同的维度上增加维度，就会导致原本的变量在行或列进行复制，从而能够得到每个token相对于其他token之间的相对位置关系。（这里具体怎么增加维度，相减，还需要仔细研究！）在做完减法之后，index最小值这里是-11。但是由于是index，所以需要保证最小的数为0，因此作者加了window_size -1 。
但这里就还剩最后一步，那就是为什么只给横坐标乘2* window_size -1呢。原因是：由于我们在计算过程中是展开计算的，因此我们在计算相对位置，也就是横坐标差值与纵坐标差值的和来确定相对位置的远近。这就会导致一个问题，那就是(1,0)和(0,1)的相对位置都是1，所以为了避免这个问题的发生，将所有的横坐标乘了一个2* window_size -1。
到这里相对位置基本就结束了。你可能还好奇相对位置偏执的shape为什么设为(2 * window_size-1)** 2。我们可以通过index来看，在做完广播减法，并且将最小的index置为0时，最大的index应该是2 * window_size-2，后来横坐标又做了乘法，因此这里最大的横坐标为(2 * window_size-2)(2 * window_size-1),纵坐标最大就是2 * window_size-2。我们将横纵坐标相加，最后就是4 * (window_size-1) * window_size。也就是最大的index可以取到它，左闭右开给index+1就得到了 (2 * window_size-1)**2。当然我不确定这么做对不对，但是我是这么理解的。
最后我们再来说一下这个self.register_buffer，就是实现相对位置index是不可学习的。
到这里就是相对位置的全部！如果没看懂我只能说很正常，要不再看看大佬写的文章。这一部分真的很难理解。
不过后面就简单多了，因为我们之前提到了shift_size==0,因此这里的mask是为None，所以我们等一下说到SW-MSA的时候再来看mask，毕竟mask是第二个很难懂的地方。不过在W-MSA时是没有的。接下来就是计算softmax，然后乘v，得到最后结果进行reshape以及dropout，linear等操作。最终输出的shape为(64,144,128)。
```ruby
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```
到这里恭喜，完成了一大部分了。return之后呢，就回到了这里
```ruby
    attn_windows = self.attn(x_windows, mask=self.attn_mask) 
    attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
    shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
    # reverse cyclic shift
    if self.shift_size > 0:
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
    else:
        x = shifted_x
    x = x.view(B, H * W, C)
    # FFN
    x = shortcut + self.drop_path(x)
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x
```
剩下的一部分，就是简单的reshape，然后进行一个window_reverse
```ruby
def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
```
window_reverse之后呢，就是reshape和前面说过的残差---ln---mlp等。到这里我们就将self.blocks中个的第一层走完了。在后半部分的shape变换是这样的：attn输出(64,144,128)--->(64,12,12,128)--->(1,96,96,128)--->(1,9216,128)。到这里我们可能就领悟到了，window_partition是将tokens切分成window，window_reverse逆操作。
接下来就是self.blocks的第二层，当然这一层shift_size就不再是0了。
```ruby
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
```
这里进来我们就又回到了熟悉的地方，之前说过的地方我们就不再赘述了，LN--->reshape，然后就是shift_szie的判断
```ruby
if self.shift_size > 0:
    shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
else:
    shifted_x = x
```
首先我们需要确定x的shape是(1,96,96,128)。这里如果你对torch.roll不太了解的话，可以百度一下，简单举个例子就是：假设有一个4 * 4 的图片，shift_size=1,那么上面代码实际上就是将dim=1维度上的最开头的元素移动到末尾；dim=2同理。
又偷了一个图，方便理解
@import "torch_roll.png"
这里作者为什么要这么做呢。这就涉及到了swin_transformer的原理思想，那就是通过移动window来实现cross Attention。但是在代码中的实现window的移动很麻烦，所以作者这里通过移动特征图来实现window的移动。这也就是我们可以看到在后面window_partition和之前是基本上没区别的。
在移动完特征图之后，就是切分window，这里和之前是一样的。然后就进入到了self.attn中。
进入之后，计算qkv的方法没有变，包括相对位置的计算也是一样的。唯一不一样的就是存在了一个mask。这个mask为什么会存在呢？这里就又涉及到了swin_transformer的原理。首先我们需要一张图
@import "mask.png"
这里我简单做了一张图，上图右侧就是我们想要得到的平移window的操作，但是在代码中并不是对window进行平移的，而是使用torch.roll对特诊图进行移动，这里就会很清晰，将编号8、6、7的window移到下面，在将编号2、5、8的window移动到右边，这样就形成了左边那张图的形状。然后再进行window的切分，就会得到左图中四个window，但是实际上我们有9个window。为了使不同window之间的计算结果不被保留，所以作者提出了mask。所以不要问什么为什么window之间的计算结果不被保留呢，这是swin_transformer设计的初衷呀：只计算window内部各个patch之间的qkv关系。
于是乎，这里的mask是怎么实现的呢？我们来看一下这里的init
```ruby
if self.shift_size > 0:
    # calculate attention mask for SW-MSA
    H, W = self.input_resolution
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    w_slices = (slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
    print(mask_windows.shape)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
else:
    attn_mask = None
```
首先这里作者生成了一个全是0的shape为(1,96,96,1)的img_mask,接下来，作者用两个循环给每个token赋上编号。首先需要了解slice这个东西是干嘛的，简单说就是slice(start,end)取出来起点到终点的位置。我们其实会发现一个问题，那就是feature map的左侧和上側的划分出的window和实际上平移后的window没有区别，真正有区别的是在特征图的最右側和最下側。因此作者这里的slice分为了三个部分
```ruby
#(slice(0, -12, None), slice(-12, -6, None), slice(-6, None, None))
```
完成编号之后，就是进到了window_partition，对window进行切分，切分完之后的shape(64,12,12,1)--->(64,144)接下来就又到了广播相减的时候啦，我真的！麻了！这里的理解就是window下，每个token和其他token做减法，结果等于0说明这两个token在相同的window下，如果不等于0，说明不在相同的window下，然后后面再进行操作。
也就是相减结果为0的，就让mask的值为0，相减结果不为0，说明不再一个window下，就让mask的值为-100。这里给的是-100。因为后面针对这个mask与attn实际上是加法操作，所以如果加了一个很负的数，再经过softmax就基本等于0啦。
```ruby
if mask is not None:
    nW = mask.shape[0]
    attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
    attn = attn.view(-1, self.num_heads, N, N)
    attn = self.softmax(attn)
else:
    attn = self.softmax(attn)
```
这里的mask在init之后，做了reshape，然后维度进行扩充，最后和attn做的加法操作，然后进行softmax得到输出结果。
到这里恭喜，我们基本上已经搞定了swin_transformer_block的内容。这一部分的主要内容就是W-MSA和SW-MSA。但是我们可能还差最后一点。
就是细心地话，你会发现，在self.blocks中末尾还有一个downsample。对，这个就是我们最后一个模块
```ruby
    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x
```
在这里self.downsample,首先这里的self.downsample什么时候才有呢，这个就要看定义了，也就是在定义self.layers时，定义了
```ruby
downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
```
看代码，可以获取巨大的信息量，首先downsample就是PatchMerging这个类，其次这个类在最后一层是不会调用的。其实我们可以先说出来它的原理，这一部分顾名思义就是patch的融合，那么我们就会得到更少的patch，对啦！所以这部分就是能够使模型成为金字塔结构的模块。
```ruby
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x
```
我们简单看一下这里patchmerging的代码。我们知道减小特征图大小就是减小H、W，通过看前向传播的代码，我们可以发现，作者将输入的特征图进行了切片截取，这样得到的x0到x3的维度都会是(1,48,48,128)，但是x输入的特征图大小是(1,96,96,128)，接下来作者将x0到x3在通道数那个维度进行拼接，最后在通道数那个维度进行linear，这样就将原先的96 * 96 个token较少到了48 * 48 个token。
整体维度变换是这样的：(1,96,96,128)--->切片(1,48,48,128)--->(1,48,48,512)--->(1.2304,512)--->(1,2304,256)最后return
这里return之后就完成了self.layers的第一层，接下来就会进入到第二层，第三层，第四层，然后head分类器，后面的步骤基本就没什么难点啦，大同小异。于是！这里swin_transformer结束~~~
最后我也只是看了一下self.layers的输入维度，至于里面就没有继续一行一行看了。整个流程都了解了，剩下的都是循环啦。
这里简单说下进入self.layers每层的维度：
(1,9216,128)
(1,2304,256)
(1,576,512)
(1,144,1024)
完美的金字塔。结束~~~

