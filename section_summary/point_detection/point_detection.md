# Point Detection
在之前努力学习了经典的点云检测算法，像pointnet、voxelne、pointpillar等。接下来我们就要在这个基础之上去掌握比较前言的技术，去了解一些比较新的点云检测算法，这篇文章呢，并不是对每个点云算法做一个非常详细的讲解，只是一个简单的介绍，后续会根据需要对点云算法进行学习。
### PDV
Point Density-Aware Voxels for LiDAR 3D Object Detection
论文链接：https://arxiv.org/pdf/2203.05662v1.pdf
基于体素的方法，点云密度感知体素网络，通过利用点中心定位和特征编码解决识别问题，直接对多类别三维目标检测的点云密度进行编码。
贡献：
Voxel Point Centroid Localization ,体素点云质心定位。PDV对每个非空体素内的点进行分区,并计算每个体素内的点质心。通过点云质心定位体素特征,PDV在第二阶段细化proposal,利用点密度分布来保留特征编码中的细粒度位置信息,避免了FPS这样的开销大的算法。
Density-Aware RoI Grid Pooling,密度感知RoI网格池化。增加了RoI网格池化将局部点密度编码为附加特征。首先采用核密度估计(Kernel Density Estimation, KDE)在每个query ball中编码局部体素特征密度。然后是网格之间的自注意力机制,并有一个新的点密度位置编码。
Density Confidence Predition,密度置信度预测。通过将最终的边框中心位置和最终边框内的原始点云数作为附加特征,进一步细化三维检测框。
### Focal Conv
Focal Sparse Convolutional Networks for 3D Object Detection
论文链接：https://arxiv.org/pdf/2204.12463.pdf
这篇文章介绍的是一个新颖巧妙的三维卷积的改进。
```ruby
#因为点云是稀疏的，不同与图像，所以在处理点云数据时，很多采用了稀疏卷积。主流3D检测网络分为常规稀疏卷积和子流形卷积。在介绍文章的贡献之前，学习下各个不同类别的稀疏卷积。在学习的过程中发现，内容还挺多的，所以准备单独写一个md文件！！
```
贡献：
提出 Focals Convolution (Focals Conv), 动态确定输入特征的输出特征的形状。焦点稀疏卷积~
提出 Focals Convlution Fusion (Focals Conv-F), 通过图像中的外观信息和大的感受野增强重要性检测。
### SST
Embracing Single Stride 3D Object Detector with Sparse Transformer
论文链接：https://arxiv.org/pdf/2112.06375.pdf
贡献：
重新思考了激光雷达的目标检测网络，通过实验发现网络步长是一个被忽视的设计因素
提出单步长稀疏变换器(Single-stride Sparse Transformer, SST)，凭借其局部注意力机制和处理稀疏数据的能力，克服了单步幅设置中的感受野收缩，同时避免较大的计算开销。
在Waymo数据集实现了先进的效果，并且在行人等微小的物体检测上有83.8%AP的精度。
### SA-SSD（后面是点和体素相结合的一些方法）
Structure Aware Single-stage 3D Object Detection from Point Cloud
论文链接：https://www4.comp.polyu.edu.hk/~cslzhang/paper/SA-SSD.pdf
关键问题：
    点云的结构信息难以建模,因为点云特征提取时的下采样操作中增强了语义信息,损失了空间信息。
    bounding box 和 classification confidences 存在不匹配的问题,边界框标记准确时分类置信度往往较低。
    用以检测三维框和分类的网络在分别输出,没有进行信息交互。
核心思想
通过深度挖掘三维物体的几何信息来提高定位精度。
    单阶段检测器 (single-stage): 将点云编码成体素特征 (voxel feature), 并用 3D CNN 直接预测物体框, 速度快但是由于点云在 CNN中被解构, 对物体的结构感知能力差, 所以精度略低.
    两阶段检测器 (two-stage): 首先用 PointNet 提取点级特征, 并利用候选区域池化点云以获得精细特征.通常能达到很高的精度但速度较慢。
two-stage的二阶段对每个RoI重新提取了空间信息。将二阶段方法独有精细回归运用在一阶段的检测方法上,为此作者在SECOND(提出了稀疏卷积,相对于密集卷积提高了速度)作为backbone基础上,添加附加任务,使得backbone具有structure aware的能力,定位更加准确。
### PV-RCNN v1、v2
点和体素结合的方法