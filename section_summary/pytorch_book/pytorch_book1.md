# PyTorch机器学习从入门到实战
随笔：前几天看到了一本书，随便翻了翻感觉有一些东西感觉印象不是很深刻。这本书就是PyTorch机器学习从入门到实战。正巧最近一直深受汽车工程学的折磨，最近因为疫情考试推迟了，才有一些空闲的时间去看这本书。《关于我用了两个小时看完一本书的事情》。所以这篇文章中有什么不对的地方，或者漏下的地方，请见谅。这篇文章主要是针对一些知识进行查漏补缺。
@import "book.jpg"
当然初次之外，通过这篇文章你还可以学习一个点，就是关于手写数字识别的项目精进，本人之前做过这个，包括在这个书中多次提到了这个项目，但是它的正确率却只有0.97左右的样子，但是反观很早之前，在我开始了解熟悉kaggle的时候，kaggle中的手写数字识别的项目，可以达到正确率百分百的人大有人在，所以我们还会针对这一点去学习。
# 书中笔记
## Numpy
关于numpy这一节，我想就不再这里过多的赘述了，后面我会针对numpy的使用，也写一个总结。
## 预训练模型加载
实际上这里和普通的模型权重加载也没什么两样，但是书中在这里提到了对预训练模型进行微处理的代码，我想可以学习一下。
```ruby
#正常的加载步骤
pretrained_state_dict  = torch.load('path of .pth')
model.load_state_dict(pretrained_state_dict,strict = True)
#如何进行微处理
pretrained_state_dict  = torch.load('path of .pth')
model_dict  = model.state_dict()
pretrained_state_dict = {k : v for k,v in pretrained_state_dict.items() if k in model_dict}#着重理解这一行
model_dict.update(pretrained_state_dict)
model.load_state_dict(pretrained_state_dict)
```
这里的微处理呢，可以看出来实际上就是将预训练权重和模型的参数进行对比，如果二者的key能够对应的上，那么就使用预训练的权重进行替换，反之则忽略。这里大家可能好奇为什么那一行代码要那么写呢，那一行长长的代码到底是什么意思呢。首先我们要清楚一点，那就是权重是个字典，所以```pretrained_state_dict = {}```是没有问题的。至于里面是什么意思，因为pretrained_state_dict是字典，里面存放的是键值对，所以做遍历，取出里面的k:v，然后判断k是否在模型参数中，如果在，那么就将这对键值对存放在新的pretrained_state_dict字典中。最后加载进去。
这样的操作一般什么时候会用呢。我的理解是，我们将别人的代码作为baseline进行了小部分的更改的时候，这个时候，有些层做了更改，有些没有，所以我们用这个操作，将没有更改的权重加载进来，更改的就忽略就好了。
## 激活函数
这一部分我打算简单过一下就好了，确定一个事情：激活函数 == 非线性函数
### Sigmoid + tanh
这个函数叫sigmoid，之前我经常打成sigmod，哈哈哈哈哈哈。
缺点：梯度消失、输出值不是0均值、反向传播求误差梯度时计算量大
tanh也存在梯度消失的问题
因此在使用中，不推荐在隐藏层使用这两个激活函数，隐藏层可以理解为神经网络的中间层。当然看很多代码，我们都会发现，sigmoid其实大多都使用在模型的最后一层，或者甚至直接放在loss中进行，比如```torch.nn.BCEWithLogitsLoss```
此外，书中还扩展了一个Hard Tanh，其将大于1的输出1，小于-1的输出-1，其他原样输出。同时这个阈值是可以更改的。
```ruby
torch.nn.Sigmoid()
torch.nn.Tanh()
torch.nn.Hardtanh(-1,1)
```
### ReLU
这是一个我们都很熟悉的激活函数了，这里有很多它的扩展，可以参考下。
### softmax
这里主要介绍softmax，这个听上去，很熟悉，但是公式总忘，每次都要查。首先softmax可以很好的应用在分类任务上，这毋庸置疑。它主要进行的就是：将一个含有任意实数的K维向量压缩到另一个K维向量中，返回的是每个互斥输出类别上的概率分布，使每个元素的范围都在(0,1)之间，并且所有元素的和为1。到这里，是不是很想知道它的公式了呢，但是我不知道怎么用md去写公司呀，只能粘个图片过来了。本来想找个公式，但是我发现了一个更好解释的图片
@import "softmax.png"
从图中很容易就明白了，假设一共有三个类别输出，我们将三个类别分别求e的指数，然后分别做分子，分母则是他们的和。就达成了我们目的。看一下简单的代码实现
```ruby
m = torch.nn.Softmax(dim = 1)
input = autograd.Variable(torch.randn(2,3))
print(m(input))
```
这里的输出最后是一个(2,3)的张量，dim = 1维度上的三个数求和为1。到这里是不是一下子明白了YOLO代码中的输出。
## Loss函数
书上是分了两部分来介绍这一块：回归问题和分类问题。针对回归问题提到了均方差损失(MSE)和平均绝对误差(MAE)，分类问题提到了铰链损失(hinge loss)和交叉熵函数(cross entropy loss)。想要看公式的话，建议去搜一搜，下面只是简单说一说。
### 回归问题
MSE均方差损失，顾名思义就是求得预测值和真实值的差值的平方求和在求均值。这种方法有一个缺点，那就是这会对异常值十分的敏感，说白了就是一失足成千古恨，毕竟利用的平方的形式。因此后来提出了绝对值的形式，那就是MAE平均绝对误差，简单来说就是将MSE的差方换成了差绝对值，同时MAE将更关注的是中位数而不是平均数。
### 分类问题
关于铰链损失，我是一点都不知道哇。铰链损失只可以解决二分类的问题。我看了很久的公式，这里只需要注意一点，那就是如果你使用的是铰链损失，那么你的0-1分布其实应该对应的是-1 和1。也就是1代表属于该类别，-1代表不属于该类别。
对于多分类问题，可以把问题转换成二分类来解决：对于区分N类的任务，每次只考虑是否属于某一个特定类别，把问题转换为N个二分类的问题。
接下来就是负对数似然函数，也就是交叉熵函数，交叉熵函数的公式，一看就懂，一用就会。
@import "ce_loss.png"
额，我没有找到不适用log来写的交叉熵函数，因为在书中提到，交叉熵损失函数可以写成一个指数相乘的形式，但是为了方便计算，我们利用对数函数把乘积变成了求和的形式。因为对数函数是单调的，取对数函数的最大化和取负对数函数的最小化是等价的，就得出了负对数似然函数，也就是交叉熵函数，如上图的形式。
### 使用
这里我们将提到一些常用损失函数的使用
```ruby
torch.nn.MSELoss()
torch.nn.L1Loss()#这个就是MAE
torch.nn.BCELoss()
torch.nn.BCEWithLogitsLoss()#在bce损失函数处理之前加了一层sigmoid
torch.nn.NLLLoss()#多分类使用这个可以注意一下。
torch.nn.CrossEntropyLoss()
```

## 梯度下降 + 优化器
这里提到了随机梯度下降、mini-batch梯度下降，这些在我之前总结的文章中有详细的提到。
同样优化器，SGD和adam等，之前也提到过。

## 手写数字识别
在这篇文章中，主要的项目实现，就是这个，但是它的正确率最高也只有0.98，所以我们这里看看kaggle中的代码实现。非常滴不行，我在kaggle上看了一下，实际能够达到百分百正确率的其实并不多。只有前四十几名能达到百分百的成绩，况且开源的代码也不是基于pytorch的
官网：https://www.kaggle.com/competitions/digit-recognizer/leaderboard
于是基于pytorch我找了两个代码，一个是最最基本的教程，一个是提交成绩能够达到0.995的代码
在第一个jupyter中呢，作者首先介绍了pytorch，后面才给出的代码。想要详细了解的可以直接去看jpy文件，这里简单说一下代码的实现。哥几个对不住，我简单看了一下这个教程，它的重点不是正确率的提高，而真的只是教程。感觉完全可以通过它的代码去入门pytorch的代码实现的整体流程。真的很详细。
那么我们就看看第二个jpy吧。
读代码还是挺简单的，原来并没有我想想的那么复杂，这个代码中使用的模型都是很简单的卷积bn、relu和池化简单的拼起来的，就可以达到0.995的效果，整体代码看下来好像没有很多令人学习的。我第一次接触这个的时候，使用的是alexnet，但是正确率却只有0.97左右，看来问题实际上应该是出在了数据增强，可以看到它的代码中使用了简单是旋转和平移
```ruby
train_dataset = MNIST_data('../input/train.csv', transform= transforms.Compose(
                            [transforms.ToPILImage(), RandomRotation(degrees=20), RandomShift(3),
                             transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]))
test_dataset = MNIST_data('../input/test.csv')
```
这好像就是代码中唯一我们可以学习的了，不对，是当时的我要学习的。当然初次之外，还有一些可以学习的，就是写代码的规范性和整个工程的规范性。值得注意的一点就是，它在对数据进行操作的时候，统计了数据集中的分布情况，这是一个很好的习惯。毕竟有时候分布不均衡也会导致模型的效果降低，其次就是数据读取的代码实际上是比较繁琐的，也是很容易出错的地方，所以它做了多次可视化。
还有代码架构很好
```ruby
n_epochs = 1
for epoch in range(n_epochs):
    train(epoch)
    evaluate(train_loader)
```
这样写的好处就是代码清晰，一目了然。
这里有一个问题就是，作者在net初始化时，写了两个for循环，是干什么用的？
```ruby
class Net(nn.Module):    
    def __init__(self):
        super(Net, self).__init__()
          
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
          
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, 10),
        )
          
        for m in self.features.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
        for m in self.classifier.children():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x     
```
我个人感觉，这里的for循环，实际上就是对权重的初始化定义。
其他好像也就没什么了，主要就是学习率和优化器的定义了,学习率的衰减。
```ruby
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
```

