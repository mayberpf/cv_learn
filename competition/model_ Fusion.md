# 模型融合
呆在家里阳不阳性我不知道，能长胖我是知道的！最近真的是一波三折，一件事挨着一件事，上周刚刚结束了汽车工程学，这周刚开始就又要开题了。。。我也很无奈，科研进度落下很多，好久都没有看过代码了，感觉自己已经不认识26个英文字母了。更何况，最近看到kaggle新出了图像分割的比赛，有关乳腺癌的一个比赛。tmd，我只有10G的3080，学校新到的服务器我好羡慕，2条3090，我选的心动款，但是跟我没多大关系了，艾，要是有2条3090我必冲银牌，但是目前看来铜牌都很难亚（根据上次的经验）。所以！言归正传，今天我又找出了上次比赛的第二名的代码。没错，就是那个将60个模型融合到一起的推理代码，我决定仔细看看模型融合是怎么实现的。
我们迅速解决，开题报告和ppt还差点东西呢。
我们依然还是一行一行的看。
```ruby
data_dir = '.'
data_dir = '../input/hubmap-organ-segmentation/'
# models_folder = 'weights'
models_folder = '../input/subweights3/'
models_folder1 = '../input/subweights4/'
models_folder2 = '../input/subweights2/'
df = pd.read_csv(path.join(data_dir, 'test.csv'))
organs = ['prostate', 'spleen', 'lung', 'kidney', 'largeintestine']
```
这里就是一些目录，目录地址文件夹中存一些文件，后面会读取的。
```ruby
def rle_encode_less_memory(img):
    pixels = img.T.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def preprocess_inputs(x):
    x = np.asarray(x, dtype='float32')
    x /= 127
    x -= 1
    return x
```
这两个函数，跟我们说的模型融合没有一毛钱关系，简单说下，第一个函数是对最后预测结果mask的一个数据转换，第二个函数是对输入数据的一个预处理。
```ruby
class TestDataset(Dataset):
    def __init__(self, df, data_dir='test_images', new_size=None):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.new_size = new_size

    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        r = self.df.iloc[idx]

        img0 = cv2.imread(path.join(self.data_dir, '{}.tiff'.format(r['id'])), cv2.IMREAD_UNCHANGED)

        orig_shape = img0.shape

        sample = {'id': r['id'], 'organ': r['organ'], 'data_source': r['data_source'], 'orig_h': orig_shape[0], 'orig_w': orig_shape[1]}

        for i in range(len(self.new_size)):

            img = cv2.resize(img0, self.new_size[i])

            img = preprocess_inputs(img)
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy()).float()

            sample['img{}'.format(i)] = img

        return sample
```
这个类就很熟悉了把，就是一个dataset。将输入的图片，以及图片的一些id，宽高等都存放在sample中。
```ruby
class ConvSilu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvSilu, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=1),
            nn.SiLU(inplace=True)
        )
    def forward(self, x):
        return self.layer(x)
```
这只是定义了一个简单的conv，实际上后面作者做了一件这样的事：融合的是60个模型，但是这60个模型的decoder实际上都是一样的，都是采用的u-net的结构，所以这里定义的卷积，实际上就是针对后面decoder进行上采样时用的卷积。
```ruby
from coat import *
class Timm_Unet(nn.Module):
    def __init__(self, name='resnet34', pretrained=True, inp_size=3, otp_size=1, decoder_filters=[32, 48, 64, 96, 128], **kwargs):
        super(Timm_Unet, self).__init__()

        if name.startswith('coat'):
            encoder = coat_lite_medium()

            encoder_filters = encoder.embed_dims
        else:
            encoder = timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=inp_size)

            encoder_filters = [f['num_chs'] for f in encoder.feature_info]

        decoder_filters = decoder_filters

        self.conv6 = ConvSilu(encoder_filters[-1], decoder_filters[-1])
        self.conv6_2 = ConvSilu(decoder_filters[-1] + encoder_filters[-2], decoder_filters[-1])
        self.conv7 = ConvSilu(decoder_filters[-1], decoder_filters[-2])
        self.conv7_2 = ConvSilu(decoder_filters[-2] + encoder_filters[-3], decoder_filters[-2])
        self.conv8 = ConvSilu(decoder_filters[-2], decoder_filters[-3])
        self.conv8_2 = ConvSilu(decoder_filters[-3] + encoder_filters[-4], decoder_filters[-3])
        self.conv9 = ConvSilu(decoder_filters[-3], decoder_filters[-4])

        if len(encoder_filters) == 4:
            self.conv9_2 = None
        else:
            self.conv9_2 = ConvSilu(decoder_filters[-4] + encoder_filters[-5], decoder_filters[-4])
        
        self.conv10 = ConvSilu(decoder_filters[-4], decoder_filters[-5])
        
        self.res = nn.Conv2d(decoder_filters[-5], otp_size, 1, stride=1, padding=0)

        self.cls =  nn.Linear(encoder_filters[-1] * 2, 5)
        self.pix_sz =  nn.Linear(encoder_filters[-1] * 2, 1)


        self.encoder = encoder


    def forward(self, x):
        batch_size, C, H, W = x.shape

        if self.conv9_2 is None:
            enc2, enc3, enc4, enc5 = self.encoder(x)
        else:
            enc1, enc2, enc3, enc4, enc5 = self.encoder(x)

        dec6 = self.conv6(F.interpolate(enc5, scale_factor=2))
        dec6 = self.conv6_2(torch.cat([dec6, enc4
                ], 1))

        dec7 = self.conv7(F.interpolate(dec6, scale_factor=2))
        dec7 = self.conv7_2(torch.cat([dec7, enc3
                ], 1))
        
        dec8 = self.conv8(F.interpolate(dec7, scale_factor=2))
        dec8 = self.conv8_2(torch.cat([dec8, enc2
                ], 1))

        dec9 = self.conv9(F.interpolate(dec8, scale_factor=2))

        if self.conv9_2 is not None:
            dec9 = self.conv9_2(torch.cat([dec9, 
                    enc1
                    ], 1))
        
        dec10 = self.conv10(dec9) # F.interpolate(dec9, scale_factor=2))

        # x1 = torch.cat([F.adaptive_avg_pool2d(enc5, output_size=1).view(batch_size, -1), 
        #                 F.adaptive_max_pool2d(enc5, output_size=1).view(batch_size, -1)], 1)
        # organ_cls = self.cls(x1)
        # pixel_size = self.pix_sz(x1)
        return self.res(dec10)
        # return self.res(dec10), organ_cls, pixel_size

```
这个就是网络模型的定义，但是实际上原始的代码不必这个要多一些东西，但是实际上并没有多大用处，所以我做了一个精简，如果你想了解他的源码：https://www.kaggle.com/code/victorsd/2nd-place-inference/notebook?scriptVersionId=106240458
我主要删除的是两个东西，一个是模型参数的初始化，一个是预训练模型的加载，这两个东西在推理过程中基本就没什么用，写出来还会影响看代码。还有一个地方，那就是模型在前向传播最后，作者得到了器官类别，和像素大小这另外两个参数，这与本次模型融合无关，并且，在后面的代码中，并不清楚作者这么做的意义是什么。
言归正传，我们首先来看作者使用的60个模型都包括什么？
```ruby
params = [
    {'size': (768, 768), 'models': [
                                    ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_768_e34_{}_best', models_folder, 1), 
                                    ('convnext_large_384_in22ft1k', 'convnext_large_384_in22ft1k_768_e37_{}_best', models_folder, 1),
                                    ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_768_e36_{}_best', models_folder, 1), 
                                    ('coat_lite_medium', 'coat_lite_medium_768_e40_{}_best', models_folder2, 3),
                                   ],
                         'pred_dir': 'test_pred_768', 'weight': 0.2},
    {'size': (1024, 1024), 'models': [
                                      ('convnext_large_384_in22ft1k', 'convnext_large_384_in22ft1k_1024_e32_{}_best', models_folder2, 1), 
                                      ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_1024_e33_{}_best', models_folder, 1),
                                      ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_1024_e38_{}_best', models_folder, 1),
                                    ('coat_lite_medium', 'coat_lite_medium_1024_e41_{}_best', models_folder, 3),
                                   ],
                         'pred_dir': 'test_pred_1024', 'weight': 0.3},
    {'size': (1472, 1472), 'models': [
                                    ('tf_efficientnet_b7_ns', 'tf_efficientnet_b7_ns_1472_e35_{}_best', models_folder, 1),
                                    ('tf_efficientnetv2_l_in21ft1k', 'tf_efficientnetv2_l_in21ft1k_1472_e39_{}_best', models_folder, 1),
                                    ('coat_lite_medium', 'coat_lite_medium_1472_e42_{}_best', models_folder2, 3),
                                   ],
                         'pred_dir': 'test_pred_1472', 'weight': 0.5},
]
```
由此可见，作者模型首先按照输入图片的大小，分为了三类。不同大小有不同的模型，主要包括：efficientnet、coat、convnext。接下来马上就到模型读取了，在下面模型读取时，如果你和我一样，不是很了解coat，那么可以先跳过那个模型，因为那个模型好像和transformer差不多。不同于其他的cnn。
```ruby
test_batch_size = 1
# amp_autocast = suppress
amp_autocast = torch.cuda.amp.autocast
half_size = True
hubmap_only = False #True #False
organ_threshold = {
    'Hubmap': {
        'kidney'        : 90,
        'prostate'      : 100,
        'largeintestine': 80,
        'spleen'        : 100,
        'lung'          : 15,
    },
    'HPA': {
        'kidney'        : 127,
        'prostate'      : 127,
        'largeintestine': 127,
        'spleen'        : 127,
        'lung'          : 25,
    },
}
```
这是一些后面可能用到的参数。
接下来，我们就要看模型融合的函数了，这个函数很长，而且里面还包含了tta，所以我打算把他分开。首先是这个函数的调用。
```ruby
for param in params:
    predict_models(param)
```
刚才我们看到了params实际上是一个列表，这里对列表进行遍历，然后扔进预测的函数。我们假设第一次遍历，那么param实际上就是size为768那个参数所在的字典传进了预测函数。接下来我们对于predict_models这个函数进行拆分。

```ruby
def predict_models(param):
    print(param)
    makedirs(param['pred_dir'], exist_ok=True)
    models = []
    test_data = TestDataset(df, path.join(data_dir, 'test_images'), new_size=[param['size']])
```
这里首先打印了传进来的参数，然后调用了dataset。这些实际上没啥用，唯一有用的就是那个models = [],这里建立了一个空的列表，应该就是用来存放后面加载的模型的。
```ruby
    test_data_loader = DataLoader(test_data, batch_size=test_batch_size, num_workers=1, shuffle=False)

    torch.cuda.empty_cache()
    gc.collect()
```
这里将dataset加载成为dataloader，没什么问题，注意后面这两行代码，都是用来释放缓存的代码。就是有时候你的代码一开始跑的还挺快，后来就越来越慢，那你就要慎重了。
```ruby
    for model_name, checkpoint_name, checkpoint_dir, model_weight in param['models']:
        for fold in range(5):
```
这两个遍历是问题的关键，首先第一个遍历我们都明白，那就是将param中model的参数取出了，包括模型名字，pth文件名和所在目录，以及该模型占所有模型的比例。最后一个参数model_weight，就是模型占总模型之间的比例。怎么理解，假如我用两个模型进行预测，一个模型的效果好，一个效果差，那么我就会让效果好的预测出来的结果，也就是mask里面的值大一点，效果差的就小一点。
第二个遍历呢，k-fold交叉验证，也就是将数据分为了5折，然后产生了5个权重，这里就需要把5个都加载进来。所以我们就可以计算，在params中有12个模型，每个模型5折，也就是作者加载了60个模型权重！！
```ruby
            model = Timm_Unet(name=model_name, pretrained=None)
```
这里就是调用了，模型的类，最后返回模型，作者在这个类中是如何调用的网络结构，实际上是使用到了timm这个库也就是timm.create_model(name, features_only=True, pretrained=pretrained, in_chans=inp_size)，我不得不说timm这个库，真的决绝子，他官方教程的开头我看了一眼，太棒了，推荐给大家。在获得模型的结构之后呢，前面说了，整个模型实际上还需要一个decoder的结构，就是u-net，我们知道u-net的结构，他需要获取encoder的各层的特征图，然后进行拼接融合。所以还需要一个参数encoder_filters，这个列表存放的就是encoder每一层计算后的特征图的通道数。剩下基本就没什么难以理解的拉。
```ruby
            snap_to_load = checkpoint_name.format(fold)
            print("=> loading checkpoint '{}'".format(snap_to_load))
            checkpoint = torch.load(path.join(checkpoint_dir, snap_to_load), map_location='cpu')
            loaded_dict = checkpoint['state_dict']
            sd = model.state_dict()
            for k in model.state_dict():
                if k in loaded_dict:
                    sd[k] = loaded_dict[k]
            loaded_dict = sd
```
这一部分就需要好好理解了，首先刚才说了，模型的权重实际上是不同模型有不同的折。举个例子tf_efficientnet_b7_ns这个模型有5折，也就是有5个pth权重文件，需要加载进来，就构成了5个model。上面代码首先获取不同折的权重文件的地址名字，然后读取。读取之后，后面这个循环实际上是在做一件事：就是将网络结构和权重文件中的结构进行对比，将能够对应上的层的参数从pth加载进来。
```ruby
            model.load_state_dict(loaded_dict)
            print("loaded checkpoint '{}' (epoch {}, best_score {})".format(snap_to_load, 
                checkpoint['epoch'], checkpoint['best_score']))
            model = model.eval().cuda()
```
在获取了整个模型的参数之后，将参数加载进来，然后model.eval()
```ruby
            models.append((model, model_weight))
```
这里就是上面初始化了models的列表，然后将模型和模型占的比例作为一个元组，添加到列表中。
```ruby
    torch.cuda.empty_cache()
    with torch.no_grad():
        for sample in tqdm(test_data_loader):
            
            ids = sample["id"].cpu().numpy()
            orig_w = sample["orig_w"].cpu().numpy()
            orig_h = sample["orig_h"].cpu().numpy()
            # pixel_size = sample["pixel_size"].cpu().numpy()
            organ = sample["organ"]
            data_source = sample["data_source"]

            
            if hubmap_only and (data_source[0] != 'Hubmap'):
                continue
```
这一块没啥意思，上面我们已经将所有模型加载进来了，之后我们会对数据进行加载，然后使用多个模型进行预测。
```ruby
            msk_preds = []
            for i in range(0, len(ids), 1):
                msk_preds.append(np.zeros((orig_h[i], orig_w[i]), dtype='float32'))
```
这里作者创建了一个预测的列表，在列表里应当存放的最终的预测值。这里做了一个循环，len(ids)怎么理解，我个人的理解就是batchsize，因为我没有做debug，所以这只是我自己的理解。因为一开始设置了test_dataloader的batchsize为1，所以这里的len(ids)实际上是1，那么就是在msk_preds中添加的是一个batch中输入图片的个数个预测值。作者首先默认添加的为均为0的输出。
```ruby
            cnt = 0
            imgs = sample["img0"].cpu().numpy()
```
第二行代码做的是将输入图片提取出来，第一行这个参数是为了计算每个模型对预测结果所占的比例。几个例子，第一个模型的权重为1，第二个也是1，那么这个cnt==1+1。后面再计算比例，就只需要用权重/cnt即可。
```ruby
            with amp_autocast():
                for _tta in range(4): #8
                    _i = _tta // 2
                    _flip = False
                    if _tta % 2 == 1:
                        _flip = True

                    if _i == 0:
                        inp = imgs.copy()
                    elif _i == 1:
                        inp = np.rot90(imgs, k=1, axes=(2,3)).copy()
                    elif _i == 2:
                        inp = np.rot90(imgs, k=2, axes=(2,3)).copy()
                    elif _i == 3:
                        inp = np.rot90(imgs, k=3, axes=(2,3)).copy()

                    if _flip:
                        inp = inp[:, :, :, ::-1].copy()

                    inp = torch.from_numpy(inp).float().cuda()                   
                    
                    torch.cuda.empty_cache()
```
这里做的操作就是传说中的tta，首先我们要知道这个东西具体是做什么，实际上就是在推理时，对图像进行数据增强，然后将该图片与数据增强后的图片一起送进模型，最后的输出结果求其平均值，即可。看上面的代码，可以看出来实际作者进行了4次数据增强，数据增强就是我们较为熟悉的旋转和对陈，还有不做增强。
```ruby
                    for model, model_weight in models:
                        out, res_cls, res_pix = model(inp)
                        msk_pred = torch.sigmoid(out).cpu().numpy()
                        
                        res_cls = torch.softmax(res_cls, dim=1).cpu().numpy()
                        res_pix = res_pix.cpu().numpy()
                        
                        if _flip:
                            msk_pred = msk_pred[:, :, :, ::-1].copy()

                        if _i == 1:
                            msk_pred = np.rot90(msk_pred, k=4-1, axes=(2,3)).copy()
                        elif _i == 2:
                            msk_pred = np.rot90(msk_pred, k=4-2, axes=(2,3)).copy()
                        elif _i == 3:
                            msk_pred = np.rot90(msk_pred, k=4-3, axes=(2,3)).copy()
```
上面说了models中存放的是各个网络模型及占比，将其取出来，然后把数据都送进去，最终得到计算结果，经过sigmoid归一化，因为之前的一些数据做了tta，也就是他可能旋转了，或者对称了，所以在最后计算结果的时候，也就是计算平均值的时候，需要将那些旋转的转回来。上面的res_cls和res_pix可以不看，没啥用。
```ruby
                        cnt += model_weight
```
这就是上面说的，计算所有模型所占比例的总和。
```ruby
                        for i in range(len(ids)):
                            msk_preds[i] += model_weight * cv2.resize(msk_pred[i, 0].astype('float32'), (orig_w[i], orig_h[i]))
```
这里就是将模型输出的结果经过resize转换成但通道图片的格式存放在预测的列表中，我们看的出来，它实际上做的是一个累加的操作！
```ruby
                    del inp
                    torch.cuda.empty_cache()
```
这两行没啥意思，直接跳过把。
```ruby
            for i in range(len(ids)):
                msk_pred = msk_preds[i] / cnt
                msk_pred = (msk_pred * 255).astype('uint8')

                print(ids[i], organ[i], res_cls[i], res_pix[i]) #pixel_size[i]

                cv2.imwrite(path.join(param['pred_dir'] , '{}.png'.format(ids[i])), msk_pred, [cv2.IMWRITE_PNG_COMPRESSION, 4])

    del models
    torch.cuda.empty_cache()
    gc.collect()
```
卧槽，终于快结束了。一开始我还想复杂了，并不需要复杂的记数来记输出了多少，我们只需要把所有的放在一起就可以了，这样记数简单得多。作者就是这么做的，作者在tta全做完之后，再求其平均值。然后看得出作者是将这个输出的图片保存了。然后再读取下一个batch的图片继续操作。
okk关于模型融合，就到这里！ 艹，我还得改ppt去。拉到把，真的每次写完这种东西，巨累