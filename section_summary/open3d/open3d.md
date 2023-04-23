# 欢迎来到点云的世界！
真不错，开始努力学习点云啦！一开始连点云的读取可视化都不会，所以这里来学习一下open3d，别问我为什么不学习pcl。
## 点云 
每个点可以包括xyz、颜色、分类值、强度值、时间等信息
存储格式：pts、LAS、PCD、xyz、asc、ply
真实物体表面的离散采样，点云处理就是从点云中提取到我们需要的信息。

明文格式.txt---直接读
pcd格式

## open3d
点云的读取和保存
```ruby
import open3d ad o3d
pcd = o3d.io.read_point_cloud()
o3d.visualization.draw(pcd)
o3d.io.write_point_cloud()
```
点云的可视化
```ruby
o3d.visualization.draw_geometries([pcd],)
```
旋转平移缩放，三维变换是点云的基础！
旋转矩阵--->九个数表示三个自由度，矩阵中的每一列表示旋转后的单位向量方向，缺点是有冗余性，不紧凑。
旋转向量--->用一个旋转轴和一个旋转角表示旋转，因为周期性，任何2nπ的旋转等价于没有旋转，具有奇异性。
欧拉角--->将旋转分解为三个分离的转角，常用在飞行器上，但因为万向锁问题，具有奇异性。
四元数--->紧凑、易于迭代、又不会出现奇异值的表示方法。

open3d和numpy相互转转换

三维数据结构

点云的常见处理方法
常见的去噪算法
1、统计滤波
```ruby

```
2、半径滤波
```ruby
pc,idx = pcd.remove_radius_outlier()
```
3、引导滤波
采样方法
1、随机采样
2、均匀采样
3、体素采样
4、基于曲率的采样
5、最远点采样
6、法向降采样方法

点云切片
包围盒
去重心操作

经过我几天的学习，我发现实际上我们对点云很少说用到上面说的一些滤波等等操作，或者说，我们对点云的操作都是基于读取之后，对一个数组进行操作。所以这里就有我们必须掌握的，像opencv一样，最常用的就是点云的读取和可视化。
这因此这里我们看下open3d和pcl是如何读取