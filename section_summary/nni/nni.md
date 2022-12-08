# NNI
最近看了有关很久之前的一个kaggle比赛的讲解视频，了解到了一个自动调参工具===NNI
这个工具实际上除了可以自动调参，还可以进行模型剪枝和架构搜索等功能，我感觉在实际应用的过程中，自动调参和模型剪枝比较适合我的使用，我看了官方的教程，这篇文章主要是针对使用NNI自动调参的工具进行的模型参数的调整的一个教程。关于模型裁剪部分，我看了一下，还有很多不是很明确的地方。
官方的github网站：https://github.com/microsoft/nni
## 安装
超级简单，我们只需要使用pip安装就可以。当然官方提供了三种安装方式，pip、源码安装、docker。这里我建议使用pip或者源码安装的方式。
```ruby
pip install nni
```
安装完成之后，我们可以通过简单的命令行来检测安装是否成功
```ruby
(torch) ktd@ktd-Alienware:~$ nnictl --version
2.10
```
注意：这里我是配置好了基础的conda环境，所以直接安装就行了，如果你没有conda环境，建议再仔细看看官方的安装文档。
## 自动调参使用教程
首先，我们还是希望大家可以去看官方的文档，官方提供了中文教程还是很贴心的：https://nni.readthedocs.io/zh/stable/
接下来我们就开始简单介绍一下如何使用nni进行自动调参。
首先，在介绍这个工具之前，我们要清楚的就是调参调的是什么？我想大家对这个都有很大的疑惑，之前在改比赛代码的时候，当时还不知道什么是自动调参，所以我就是手动改参数，然后再训练，全部参数的更改都是凭感觉进行修改，实际上这个过程就是自动调参的一个过程。但是当时真的不清楚参数都可以包括什么，所以只会更改阈值，要不然就是更改图像的分检测辨率。当时甚至连学习率都不敢改，实际上就是这样的，就是我们如果对一个代码不是充分的了解，那么可能对于参数的了解，就不会那么全面。所以还是鼓励大家去debug代码，最起码真的能够进步很快。
好了，说的太多了，我们来说说这里的参数可以包括什么，其实会包括很多：（总结一句，就是但凡是模型在一次训练中需要的参数都是我们可以更改的）我们可以从不同的部分开始，一举例子大家就能明白了。首先数据部分：令人熟知的batchszie，还有可能是数据在训练时发生数据增强的概率，数据增强图像旋转的角度等、关于网络架构：可能是一些通道数的更改、过于优化器：可能是优化器的类型，学习率的大小，动量的大小，学习率的衰减；关于后处理：可能是iou的阈值等、是不是感觉，能够设置的参数太多了。
接下里我们就举一个简单的例子，从头开始实现自动调参。
这个例子是基于手写数字识别的一个模型的调参，设置的可调的参数有学习率，动量和模型通道数。
首先我们需要来看看最基本的模型
```ruby
import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
#======这里即我们需要调节的参数========#
params = {
    'features': 512,
    'lr': 0.001,
    'momentum': 0,
}
#====================================#
optimized_params = nni.get_next_parameter()
#这里我们并没有运行nni，直接调用nni.get_next_parameter()会返回一个空的dict
params.update(optimized_params)
print(params)

training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

batch_size = 64

train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, params['features']),
            nn.ReLU(),
            nn.Linear(params['features'], params['features']),
            nn.ReLU(),
            nn.Linear(params['features'], 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    accuracy = test(test_dataloader, model, loss_fn)
    nni.report_intermediate_result(accuracy)
nni.report_final_result(accuracy)
```
代码是十分的简单，有一点可以注意一下，就是代码中的运行应该需要下载数据集，但是我在运行时，下不下来，就会报错，这里建议大家可以看报错，然后在浏览器打开网址下载数据集，然后放在对应的文件夹即可。其实现在看来，上面的代码实际上和最普通的代码有一定差别，差别就在于下面这几行
```ruby
optimized_params = nni.get_next_parameter()
self.linear_relu_stack = nn.Sequential(
    nn.Linear(28*28, params['features']),
    nn.ReLU(),
    nn.Linear(params['features'], params['features']),
    nn.ReLU(),
    nn.Linear(params['features'], 10)
)
optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], momentum=params['momentum'])
nni.report_intermediate_result(accuracy)
nni.report_final_result(accuracy)
```
看到这里基本就明白了，nni自动调参的工作原理很简单，首先我们需要设定需要调整的参数，然后在一开始就使用nni调用这些参数，然后在后面的代码中所有这些参数都使用params来替代，最终我们需要设定一个评价指标来反馈给nni，让nni知道这一套参数怎么样，然后就可以合理规划下一步怎么调整参数了。
ok，接下来就是如何调用这个nni了，这个部分可以通过简单的Python文件运行，也可以使用命令行，首先我们介绍下使用python的方式,我们将按照每一部分进行介绍
```ruby
search_space = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}
```
这一部分叫做搜索空间的定义，实际上呢，就是我们需要告诉nni，每个参数可以调整的范围，这里有不同的取值方式：choice、loguniform、uniform，分别代表features会在[128,256,512,1024]中取值，lr会在0.0001到0.1之间，取值符合指数分布，momentum会在0到1之间。
```ruby
from nni.experiment import Experiment
experiment = Experiment('local')
```
这里是配置实验，可以理解为实例化，并且确定在本地实验。
```ruby
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'
```
首先在清楚一个概念，那就是nni测试一组参数的整个过程叫做一个trial。所以这里trial_command指的就是我们运行训练文件的命令代码。这里的trial_code_directory表示的训练文件相对本文件的相对路径，感觉为了方便，一般都会把这个和训练文件放一起吧。
```ruby
experiment.config.search_space = search_space
```
这里是配置搜索空间，基本不用改。
```ruby
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
```
这里就是配置nni调参的算法，也就是优化调参的方法，当然还有其他的不同调优算法，详情可以看一下官方文档。
```ruby
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
```
这里max_trial_number设置的是进行trial的次数，trial_concurrency表示的是同时进行trial的个数，当然我们也可以设置进行trial的最大时间max_expreiment_duration
注意：这里的max_trial_number设置为10，是为了能够快速运行且结束，但是在实际的应用过程中，我们应该设置更大的数值。
```ruby
experiment.run(8080)
done = input("实验结束，注意查看。")
```
最后这里就是运行实验，8080代表的是端口号，这样我们就可以通过网页控制台的方式查看效果：http://localhost:8080
相信你肯定注意到了，为什么后面加了一个input指令，因为我们的代码在运行结束后，会自动的补全并且运行```experiment.stop()```，但是一旦结束，我们就不能通过网页控制台看效果了。因此在最后加入了一句input指令。
到此为止，我们就可以在网页中看处理效果了。
@import "1.png"
@import "2.png"
就并不过多的描述这个网页控制台了，主要分为两个部分：overview、trial detail。详细的大家可以去看看，最后我们可以通过上传的指标进行评价，在上述的代码中，我们上传的是正确率，那么就可以选择正确率最高的一组参数作为我们的超参数。
到这里，我想你一定好奇，我们的训练代码训练一次要好几个小时，有时候甚至一晚上，那这里一个trial训练一次，那不得跑上十天半个月。这就涉及到了，实际上我们在进行trial的时候，我们需要将epoch调小，没必要必须和正常训练一样。还有一点就是，数据集如果很大的话，其实我们也可以将数据量减半，这都是可以的。
okk，恭喜你，已经成功的入门了。我个人感觉哈，因为很多代码都没有这么写的格式，所以想要用到自己的代码的话，还需要一个实践。
最后我们简单看一下命令行的格式，这一部分，训练代码是不变的，我们需要做的就是将调用nni的Python代码转换为一个config.yaml文件，然后通过命令行来运行就可以了。
```ruby
search_space:
  features:
    _type: choice
    _value: [ 128, 256, 512, 1024 ]
  lr:
    _type: loguniform
    _value: [ 0.0001, 0.1 ]
  momentum:
    _type: uniform
    _value: [ 0, 1 ]

trial_command: python model.py
trial_code_directory: .

trial_concurrency: 2
max_trial_number: 10

tuner:
  name: TPE
  class_args:
    optimize_mode: maximize

training_service:
  platform: local
```
有没有感觉这样会比Python简单，其中包括的东西都是一样的，具体含义大家可以看官方的文档。阅读的能力真的很重要！
然后运行的命令行
```ruby
nnictl create --config config.yaml --port 8080
```
就是酱紫简单！