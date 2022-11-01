# dataset
要点：
1、classdataset类继承torch.utils.data.dataset
2、classdataset的作用是将任意格式的数据，通过读取、预处理或数据增强后以tensor的形式输出，其中任意格式的数据可能是以文件夹名作为类别的形式、或以txt文件存储图片地址的形式、或视频、或十几帧图像作为一份样本的形式。而输出则指的是经过处理后的一个batch的tensor格式数据和对应标签。
3、calssdataset主要有三个函数完成：__init__,__getitem__,__len__。
#### __init__函数
主要是完成两个静态变量的赋值。一个用于存储所有数据路径的变量，变量的每个元素即为一份训练样本，（注：如果一份样本是十几帧图像，则变量每个元素存储的是这十几帧图像的路径），可以命名为self.flienames。一个用于存储与数据路径变量一一对应的标签变量，可以命名为self.labels。
#### __getitem__函数
根据索引返回对应的数据，这个索引是在训练前通过dataloader切片获得的。获取数据后，对数据进行预处理，数据的预处理主要是通过torchvision.transforms来完成
具体参考官方文档：https://pytorch.org/vision/stable/transforms.html

当然数据增强不仅仅限制与这个库里的，我们也可以自己设计数据增强的方法，例如在data_transform中介绍过的randomerasing等操作。想要将其添加到dataset中，只需要将其添加到原始定义的transform_lists中即可。
####__len__函数
返回数据长度，也就是样本总量。

#### 验证classdataset
```r
train_dataset = My_Dataset(data_folder = data_folder)
train_loader = DataLoader(train_dataset,batch_size = 16,shuffle = False)
print('there are total %s batches for train'%(len(train_loader)))

for i ,(data,label) in enumerate (train_loader):
    print(data.szie(),label.size())
```
#### 分布式训练的数据加载方式
如果是多卡，但是为了高速高效读取，每张卡上也会保存所有的数据信息，也就是self.files 和 self.labels的信息。知识在DistributedSampler中会给每张卡分配互不交叉的索引，然后由torch.utils.data.DataLoader来加载
```r
train_dataset = My_Dataset(data_folder = data_folder)
sampler = DistributedSampler(dataset) if is_distributed else None
train_loader = DataLoader(train_dataset,batch_size = 16,shuffle = (sampler is None),sampler = sampler))
```
#### 完整流程
1、初始化
2、getitem
3、索引是通过dataloader获取的
4、单卡索引可以通过shuffle设置是否打乱。但是在多卡中，shuffle是由distributedsampler来完成的，因此shuffle与sampler只能有一个，另一个必须是none

#### 超大数据集加载思路
思路：
将切片获取索引的步骤放到 classdataset初始化的位置，此时每张卡都是保存不同的数据子集，通过这种方式，可以将内存用量减少到原来的world_size倍(world_size指卡的数量)
具体参考：https://zhuanlan.zhihu.com/p/357809861