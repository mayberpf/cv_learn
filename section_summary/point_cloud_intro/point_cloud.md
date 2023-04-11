# Point_cloud
本篇文章介绍一些三维点云的基础知识，是做点云深度学习之前的必修课。
### python 点云可视化
关于点云的可视化，这里有三种方案：mayavi、matplotlib、cloudcompare
其中matplotlib没必要再说了，因为之前用它做过三维车道线检测点的可视化，所以对这个稍有了解。其实不论是mayavi还是matlotlib都是将点云利用numpy进行读取，然后取出xyz的坐标，然后利用对应的函数进行可视化。
这里主要介绍的一个工具是cloudcompare，这个真的很适合初学者，不需要任何代码，你只要到官网搜索这个软件安装就好啦。然后呢，我们不仅可以可视化点云，还可以选取点云中的某一个点，然后得到它的xyz数据。
### 点云存储形式
常见的点云存储形式有：pcd、ply、txt、bin等文件。其中的基本组成大体是相同的，主要包括：xyz点云的空间坐标、i强度值、rgb色彩信息、a代表的透明度、nx、ny、nz代表的是点云的法向量。但是一般一个点云数据不会全部包括，但是一定会有xyz的坐标，其他都不是必要的啦。
首先我们先来看pcd文件
这里需要声明一件事===pcd文件的存储方式有两种：二进制和ascii，ascii存储的数据是可以用文本编辑器打开的，但是二进制就不行，同时二进制的也不能通过cloudcompare进行可视化。
然后我们来说一下pcd文件的内容，分为文件说明和点云数据两部分
```ruby
# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7#版本
FIELDS x y z#每个点组成
SIZE 4 4 4#每个数据占用个的字节数
TYPE F F F#数据对应的类型F为浮点数U为无符号整型I标识整型
COUNT 1 1 1#数据对应维度
WIDTH 57600#点的数量
HEIGHT 1#无序点云默认为1
VIEWPOINT 0 0 0 1 0 0 0#点云获取的视点，用于坐标转换
POINTS 57600#点的数量
DATA ascii#存储类型
```
后面点云数据就是我们熟悉的xyz。
txt文件，典型的就是modelnet40，其中一个点6个数据：xyz和nx、ny、nz法向量。
至于读取，就相当与文件的读取数组一样。
```ruby
def pcd_read(file_path):
lines = []
with open(file_path, 'r') as f:
lines = f.readlines()
return lines
file_path = 'rabbit.pcd'
points = pcd_read(file_path)
for p in points[:15]:
print(p)
```