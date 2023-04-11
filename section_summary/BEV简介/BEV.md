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
 