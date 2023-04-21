# OpenMMLab教程
核心组件为MMCV
针对检测任务分为2D和3D的MMDetection、MMDetection3D，针对图像分割的MMSegmentation、针对目标跟踪的MMTracking等，基于pytorch和mmcv
### mmdetection安装
这一块暂时没有做好！我是按照一篇文章的步骤来做的，但是安装完成之后，虽然能够运行，但是只限于在cpu上面运行，我真的乌鱼子，下面是它的一个安装步骤，实际上来说还是很简单的。
```ruby
conda create --name mmdetection python=3.8 -y
conda activate mmdetection 
conda install pytorch torchvision -c pytorch
pip install -U openmim
mim install mmcv-full 
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection 
pip install -v -e .
#但是在做验证，跑demo的时候，有一些问题，所以就去看了下官网的教程
#可能还需要这两个命令
mim install mmengine
mim install "mmcv>=2.0.0
```
接下来就是跑一个demo验证安装是否成功
```ruby
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```
然后成功的话，就能够在output里面发现可视化的检测结果。
但是有一个致命的问题，那就是
```ruby
Python 3.9.12 (main, Jun  1 2022, 11:38:51) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> print(torch.cuda.is_available())
False
```
对，没错，很尴尬，但是到这里就只能先这样啦。回头再说。