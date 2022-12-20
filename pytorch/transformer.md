# Pytorch搭建Transformer网络
首先是我们都比较熟悉的VIT的模型图
@import "transformer.png"
也就是说：对于一张图片而言，先将其分割为N*N个patch，然后将patch进行Flatten，再通过一个全连接层映射成tokens，对每一个tokens加入位置编码（position embedding），会随机初始化一个tokens，concate到通过图像生成的tokens后，再经过transformer的encoder模块，经过多层encoder后，取出最后的tokens（即随机初始化的tokens），再通过全连接层作为分类网络进行分类。
#### 分块
目前可以通过两种方式实现分块，一种是直接分割，一种是通过卷积核和步长都为patch大小的卷积来分割。
##### 直接分割
直接分割即把图像直接分割成多块。代码中实现需要使用einops这个库，完成的操作是将(B,C,H,W)的shape调整为(B,(H/P * W/P) , P* P*C)
```r
from einops import rearrange , repeat
from einops.layers.torch import Rearrange
self.to_patch_embedding = nn.Sequential(
    Rearrange('b c (h p 1)(w p2)->b (h w)(p1 p2 c)' , p1 = patch_height, p2 = patch_width),
    nn.Linear(patch_dim,dim)
)
```
Rearrange用于对张量的维度进行重新变换排序，可用于替换pytorch中的reshape，view，transpose和permute等操作
这个函数我也没有了解过，所以就搜了一下，有一个参数是映射关系，真的可以这样写呀。
```r
def rearrange(inputs, pattern, **axes_lengths)⟶ \longrightarrow⟶ transform_inputs

inputs (tensor): 表示输入的张量
pattern (str): 表示张量维度变换的映射关系
**axes_lengths: 表示按照指定的规格形式进行变换
#假设images的shape为[32,200,400,3]
#实现view和reshape的功能
Rearrange(images,'b h w c -> (b h) w c')#shape变为（32*200, 400, 3）
#实现permute的功能
Rearrange(images, 'b h w c -> b c h w')#shape变为（32, 3, 200, 400）
#实现这几个都很难实现的功能
Rearrange(images, 'b h w c -> (b c w) h')#shape变为（32*3*400, 200）
```
##### 卷积分割
卷积分割比较简单，使用卷积核和步长都为patch大小的卷积对图像进行一次卷积就可以了
```r
self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
```
在swin transformer中就是使用的这种卷积分块方式，注意卷积之后没有再加全连接层
#### Position Embedding
正弦编码方式只适用于语音、文字等1维数据，图像是高度结构化的数据，用正弦不合适
在ViT和swin transformer中都是直接随机初始化一组与tokens同shape的可学习参数，与tokens相加，即完成了absolute position embedding
在ViT的实现方法
```r
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
x += self.pos_embedding[:, :(n + 1)]
#之所以是n+1，是因为ViT中选择随机初始化一个class token，与分块得到的tokens拼接。所以patches的数量为num_patches+1。
```
在swin transformer 中的实现方式
```r
from timm.models.layers import trunc_normal_
self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
trunc_normal_(self.absolute_pos_embed, std=.02)
```
就先写到这里，我感觉这样看是看不懂的，实际上我们要到代码上去调试，才能方便我们理解。
参考：http://t.csdn.cn/VUzSd