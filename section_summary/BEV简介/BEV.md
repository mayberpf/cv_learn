# BEV感知

BEV视图的优势：尺度变化小，遮挡小
BEV感知：客观视图在BEV视图下的表征
多传感器输入：图像、激光雷达、毫米波雷达

数据形式
相机图像、雷达点云
单独图像、单独点云、图像+点云
图像优势：纹理丰富、成本低
基于图像的任务、基础模型相对成熟和完善，比较容易扩展到BEV感知算法中。（H，W，3）、（x，y）
BEVFormer--->多视角图片--->backbone--->多视角图片特征
点云：基本组成单元是点，集合叫做点云
点云的特点：稀疏性、无序性、是一种3D表征、远少近多

BEV数据集介绍---KITTI
KITTI只提供了图片视角下的目标检测框 
nuscenes
6个相机--多视角数据
maps
samples关键帧
sweeps未标注的数据
v1.0-标注文件


感知方法分类
BEV LiDAR
输入点云--输出检测
pre-BEV先提取特征再生成BEV特征===pv-rcnn
post-BEV 先转换到BEV视图，提取特征===pointpillar

BEV Camera
2d backbone --->view transformation(3D到2D或者2D到3D)--->检测头
利用多视角的2d图像到3d世界！最终得到俯视图
视角转换是最重要的，可以成为创新点。

BEV Fusion

BEV感知算法的优劣
纯视觉60左右比纯点云低10个点左右。

BEV感知算法的应用
课程框架介绍与配置

## 第二章
核心围绕转换模块，检测、分割===BEV中
基础模块补充讲解
2D图像处理：图像处理网络基本组成、网络层数越深越好吗？考虑梯度消失（学着学着把最开始的忘了）---残差特征（梯度保留）！
3D特征之点处理方案：point-base、voxel-base
### 2d到3d的转换模块
动机：基于采集到的环视图像信息去构建BEV视角下的特征完成自动驾驶感知的相关任务。即：相机视角向BEV视角下的转变
媒介：3D场景
2D--->BEV
2D--->3D--->(高度压缩)BEV
LSS算法
深度估计分布---Lift
Pseudo LiDAR

### 3D到2D的转换模块
DETR3D
Explicit Mapping
Implicit Mapping

### BEV感知中的transformer
transformer---注意力机制
空间注意力STN
在空间中捕获重要区域特征
通道注意力SENet
在通道中捕获重要区域特征
混合注意力CBAM
在空间和通道均进行注意力增强（串行）
如果先后顺序发生改变呢？
如果是并行？？？
 transformer自注意力机制