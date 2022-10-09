# HuBMAP
官网：https://www.kaggle.com/competitions/hubmap-organ-segmentation
#### story==故事
很多人说写论文，就是写故事，那我想如果真的那么简单，那么我一定可以写的很好。
其实我现在一直很后悔一件事，那就是今年暑假我并没有去参加肠胃分割那个比赛，因为这两场比赛都是关于分割的，而且时间正好是一个结束一个开始，我透！如果当初不错过，这次也许就能拿牌了，惨兮兮。0.75967这是我成绩，铜牌线是0.76254。我可能想先讲个故事，在这个故事里是我参加这个比赛是各种想法，后面正文才是我对于一些知识整理。
毕竟是第一次参加这个比赛，一开始我真的满怀希望，况且我还报了个**之眼的比赛班，说实话，我当初想的就是，就算我可能不太行，但是这花钱报了班，它总会给代码，我总不至于成绩很差，也就可以把这段经历写在简历上。后来才知道这都是我一个人在yy。虽然成绩不是很好，但是我想我还是可以写在简历上的，哈哈。
那就从比赛刚开始说起，其实比赛很早就开始了，但是一开始没怎么有人呀，它这个比赛班也是在比赛还剩最后一个多月的时候开班的。其实这一点我也感觉到了，没必要说比赛一开始就去专注比赛，一个多月就够了，甚至有些厉害的人几个星期就很够了。然后我就上了第一节课，针对这个图像分割的比赛呢，（因为我是第一次参加比赛嘛，所以我就只是跟随比赛班的任务去做的，包括每节课布置的作业），一开始我们使用的就是经典的resnet模型，印象中当时我还在家，它的代码让我看的并不是那么简单，包括一些k-fold，还有模型为什么会有resnet50和resnet101等，我简单的看了代码，我不理解，但是我没有第一时间去debug，确实在家不是很方便，我相信绝大部分人拿到代码，第一件事就是去run，然后submit。况且我第一次参加，我还不会怎么submit。就这样我提交了自己的第一次。就是在去研究院的前一天，提交后lb400名，我就知道了路还很长。
对了，当时有一个作业实际上是替换模型，因为用到了resnet50和resnet101，作业是替换resnet50为efficientnet。然后我一脸懵逼？什么替换网络，我们都知道，替换模型很简单,把这个net替换了不叫好了。
```r
model = net()
```
问题是什么，我要不要考虑输入输出的shape等等，就我们大家都知道一个人第一次做一件事，总会感觉无从下手。所以这件事其实我拖了很久。就是我一直在期待老师可以发一个代码，就是它替换好的代码是什么样的，这样我就可以对比一下，它是改了哪里。但是不幸的就是这件事没有，不可能，我才知道真的不会有人手把手教你，这个时候你才能体会到知识的价值。当然当时没做这件事还有一个原因：代码我都还不是很清晰，我好想没有看过这种k-fold，以及模型融合，还有tta等这些操作是干什么的，我不懂。所以我做的第一件事就是我要去debug。我相信，我debug之后了解了整个模型之后，我才能具备替换网络模型的能力。事实证明，就是这样的。
再后来我会更换网络模型之后，并且了解那一套代码之后，我就像一个小孩子一样，开始疯狂尝试，resnet50、resnet101、efficientnet各种融合，但是印象中分数最多也就是提升0.01而已。于是关于这个模型就止步于此了，但是我看讨论区里有人真的就使用efficientnet拿到了一个0.67左右的成绩，但是我却达不到，可能是因为使用的是efficientnet的小模型吧，就是b2，可能他们是使用的b7。(今天可能只能写story的部分了，关于后面的部分要等到明天了)所以他怎么做的，只能明天看了。之所以当时没看呢，是因为比赛班后来推荐我们使用swin transformer+upernet的一个模型，然后我明显感到就是我的电脑针对这个模型进行训练确实不太行，我渴望一张3090，后来有了一张3080才让我看到了一点点希望。是这样的，但是很明显batch_size到后面也是不行了。实际上呢，为什么比赛班会推荐这个模型，我基本已经get到了，实际上就是好像大家都是跟随青蛙哥（就是打比赛很厉害的一个人）的建议去做的。于是很简单，我就去跑了这个模型，然后成绩非但没有提高，而且还低的难以置信，我陷入了沉思，后来怎么说呢，错误在，我们在推理完之后，需要将mask进行一步resize，那么推理的数据的大小实际上并不是固定的，而我的代码中认为它是固定的（总结一下，这个错很zz）。再后来呢，我把输出的size改了之后呢，成绩依然不是很好，于是乎我开始调试阈值，因为我好想确实不太清楚应该调什么。在调了阈值之后呢，成绩终于有了提高，但是我不知道哇，为什么没有老师说的那么厉害，感觉我的提交成绩不太行呀。后来呢，因为课上老师说的是替换一下decoder，替换为segformer，对，其实很简单，我换了之后，提交成绩也不是很好，就在这里我彻底陷入了迷茫。后来也是看到讨论区有人说，resize的大小会提高一两个点，我起初是尝试直接改推理的代码，很明显这不切实际。后来我改训练代码，重新训练，但是问题来了，我的显存大小真的太小了，只有10G不太行，显存很容易就炸了。在这之间呢，。。。我破防了！刚才我打开了我的提交，出问题了呀！我选择了一个公榜成绩最高的作为提交成绩，但是这不是我私榜最高的成绩！而我私榜最高的成绩可以达到铜牌的的成绩，我透！！
@import "score.png" 
#### 我只能说确实实力还是不行。
那我们继续吧，后来就是我不清楚为什么换了segformer的decoder，但是成绩却仍然不好，最后我换了一个模型也就是mitB2+segformer的模型，这个效果可就比之前的模型好太多了，也就是换了这个模型我开始上了0.7。然后就是各种调参，包括什么阈值呀，还有resize的大小呀，以及resize的mode，以及loss函数，以及推理时的tta，推理时进行的模型权重占比的设置。我印象中那个时候公榜成绩很不错了，后来就是我又尝试了新的模型coat，我也是这么开始一点点了解有关transformer的东西。但是问题来了，coat我记得当时的问题是，模型收敛很慢，而且我电脑训练的也慢，最后感觉收敛很久，成绩都上不去。后来就是通过训练大图片，一点点提高成绩，但是我的显存不大，从768到896再到1024，我的显存就炸了。再后来呢就基本没怎么做了，原因呢首先是我感觉有点无从下手了，不太会做了。其次就是课题组有任务，我可能要去做了。但是其实我还是有一点想法，就是我知道可以把多个模型融合起来，但是确实有点无从下手，还有就是针对器官的种类进行训练，因为针对肺的推理成绩不太理想，所以可以针对肺使用一个模型进行训练，然后单独做推理，以及在讨论区可以看到就是针对不同器官的类别使用不同的阈值。其实这些我都懂，但是就是感觉无从下手，说的再简单点，就是我怎么用代码把我的想法实现是我们每个人都会面临的巨大问题。再后来就是等待比赛结束，真的太卷了，就是越到后面越卷，起初就是如果你一天没有提高，那么你的名词就会下降5名，越到后面呢。。。一言难尽。但是其实我们想到，最后成绩出来，我的私榜成绩会比公榜成绩高那么多名词，可能这就是k-fold以及tta的魅力把。好啦，我想故事就讲到这里的，明天就要去讨论区看看了，看看前面的code都是怎么做的。



#### 正文
首先是一张总结的思维导图
@import "思维导图.png"
##### 数据预处理
对于这次比赛，数据预处理主要针对一下几件事情：
1、原始图片数据集是3000 *3000的 tiff的格式，需要提交的csv文件中需要将预测的mask图片转成rle编码。所以针对原始图片有切patch的。但是后面发现切patch的效果没有不切的好。所以到后面就不采用切patch的方法，而是使用resize的方法
2、比赛进行到后面，也有人提供了external data。那个时候我真的已经没怎么关注了这个比赛了。
在看了前几名的代码之后:没有人去分patch。原因可能是因为3000 * 3000并不是很大。
第二名使用了external data
##### 数据增强，dataset，DataLoade于二分类和多类分割的都用了这个！
1、使用albumentations库，这个库的官网：https://albumentations.ai/docs/ 从代码中可以看出来，这里使用的数据增强按照train和val分为了两部分，train部分从resize和horizontalFlip选一个进行。只是一开始使用这个库的数据增强，后来就没有使用了，而是使用青蛙哥写的数据增强的代码。
```r
import albumentation as A
def build_transforms(CFG):
    data_transforms = {
        "train": A.Compose([
            A.OneOf([
                A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST, p=1.0),
            ], p=1),
            A.HorizontalFlip(p=0.5),
            ], p=1.0),
        "valid_test": A.Compose([
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            ], p=1.0)
        }
    return data_transforms
```
2、多分辨率训练的方案
因为类别说比较少，这个比赛只有五类，所以在这个比赛中work的效果不是很明显。
我找了一下代码，发现一个问题，就是实际上在我用的代码中没有使用到多分辨率训练的方案。
3、色彩迁移：根据目标图片调整图片的颜色配准
color transfering
vahadane normaization
传统图像处理？配准？具体了解要去看论文和前几名的代码
第二名
general：random cropping/padding , scaling , rotating , flipping , color change,亮度增强，随机噪声增强
第四名
##### model
1、这里有一个库segmentation_models_pytorch
预训练---完形填空===transformer的预训练BERT如何无损的迁移到cv中。
2、在比赛中还有人使用了mmsegmentation和Detectron等
因为对框架的不够熟悉，所以没有使用。
3、个人在比赛中对模型的选择调整
resnet+unet
将resnet换为提取特征能力更强的efficientnet-b0到b7（在模型中我只使用了b2，但是肯定是模型越大精确度越高的）
efficientnet+unet
但是特征能力仍然不足，所以在efficientnet和unet中间插入FPN结构，为了让每个pexi学到的感受野更大，在网络模型的最后加入了ASPP的模块
efficientnet+FPN+unet+ASPP
为进一步提高特征的提取，重新替换encoder和decoder
Swin+upernet
将upernet替换为基于transformer的分割模型segformer
swin+segformer
理论上这个模型的效果应该很好，但是在我实验中，实际上效果并不是很好，因此我将encoder部分换成了mit-b2
mit_b2+segformer
前面我们先不考虑名词靠前的人的代码是怎么做的，放到最后我们在看模型代码的。
##### loss
1、使用segmentation_models_pytorch库中的BCEWithLogitsLoss和Dice和TverskyLoss
2、更改loss函数为bce+dice，从代码中看，还有一个loss函数是aux_loss，这好像是一个辅助损失，实际上我们损失函数一般都是针对输出计算损失，然后反向传播的。但是这个损失应该是针对某一feature map进行反向传播。有关bce损失函数就说我们熟知的交叉熵损失函数，但是```torch.nn.functional.binary_cross_entropy_with_logits```和```torch.nn.BCEWithLogitsLoss```有啥区别呢？但是我们其实知道就是BCELoss和BCEWithLogitsLoss的区别就是多了一个sigmod的问题。针对Dice损失函数，好像pytorch中没有封装dice损失这个函数吧，但是在segmentation_models_pytorch这个库中实际上是可以调用dice_loss的。
```ruby
import torch.nn.functional as F
import segmentation_models_pytorch as smp
Dice_loss = smp.losses.DiceLoss(mode='binary')
#在net中调用损失函数，这里省略了一些细节
class Net(nn.Module):
	def load_pretrain( self,):
        pass
	def __init__( self,):
		super(Net, self).__init__()
        pass
	def forward(self, batch):
        output = {}
        if 'loss' in self.output_type:
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit,batch['mask'])
            output['dice_loss'] = Dice_loss(logit,batch['mask'])
            for i in range(4):
                output['aux%d_loss'%i] = criterion_aux_loss(self.aux[i](encoder[i]),batch['mask'])
        if 'inference' in self.output_type:
            output['probability'] = torch.sigmoid(logit)
        return output

def criterion_aux_loss(logit, mask):
	mask = F.interpolate(mask,size=logit.shape[-2:], mode='nearest')
	loss = F.binary_cross_entropy_with_logits(logit,mask)
	return loss
#训练函数中调用模型，并且提取损失函数，将损失函数求和并做反向传播
output = net(batch)#这里的output只有loss
loss0  = output['bce_loss'].mean()
loss1  = output['aux2_loss'].mean()
loss2 = output['dice_loss'].mean()

optimizer.zero_grad()
scaler.scale(0.5*loss0+0.2*loss1+0.5*loss2+0.2*loss1).backward()#============这里改了loss函数
```
3、不平衡处理？
##### metric
1、dice
2、IOU？
##### train
1、混合精度训练
2、单机多卡训练
##### val&test
1、result visual于二分类和多类分割的
2、K-fold Cross Validation
3、Model Ensemble
4、Test Time Augmentation
5、Pseudo label
第四名
第二名
##### 结果后处理
并没有分patch----因为分辨率只有3000*3000
但是一开始我们做了切patch
数据 编码的方式---numpy2rle
第二名用了external data 但是应该美起到什么作用

transformer，dataset，dataloader
albumentations 库
手写的数据增强---
色彩迁移：根据目标图片调整图片的颜色特征   hsv

这里没有做

lib：segmentation_model_pytorch as smp


timm库
第二名的代码要看？？？60个模型？

第四名
对于难识别的类
没开元？

使用框架

自己用了非常重的encoder

loss---nn.bce
diceloss

第二名
px_size
目标检测
多任务loss

metric
dice，iou

train_one_epoch

test_one_epoch
trick
    k-fold
    model-ensemble
    tta
    pseudo label
    stain normalization

后处理算法
    CRF
    分水岭


简历书写
数据集--->transformer--->
数据集是什么+增强+模型+loss+推理+指标
+个人博客地址
github repo

技能特长
每个项目都要有描述自己做了什么
数据集有多少张，最终的metric多少

cutmix？？
目标检测很有用？
写一些你懂的，有把握的，不要写有歧义的
cutout？？？

