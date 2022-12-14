# 33
# 图像几何变换的基本步骤
我们使用几何变换来修改图像像素的空间排布，其主要包括两种基础操作：
    坐标空间变换：这对应着变换前后像素坐标的映射关系，即对应坐标的调整。二维仿射变换关键的性质是保留点、直线和平面
    赋予新像素值：对新位置进行赋值，需要借助于像素插值的方法来完成

# 仿射变换包括哪些？
主要包括缩放scale、旋转rotate、平移translate、剪切sheer

# 仿射变换的统一形式？
利用齐次坐标的形式来进一步简化概念，通过增加一个额外的维度w后，可以用来对几何体进行缩放旋转平移透视投影的矩阵变换。
仿射变换可以写成3 * 3方阵的形式
@import "仿射变换.png"

# 放射变换统一形式的优势是什么？
使用这种同一形式来表示的一个显著优势在与这提供了把一系列变换连接在一起的框架。实际处理中，可以通过将所有的变换按照执行顺序依次级联相乘后，再直接作用在原图上。

# 图像几何变换的不同构建方式的具体方式及优缺点？
变换的映射过程实际上有两种构建方式:
    正向映射(forward mapping)(x', y') = A(x, y):直接对每个输入图像上的像素位置(x,y)使用坐标关系(例如仿射变换中的变换矩阵A)计算出输出图像上相应像素的空间位置(x',y')。
    反向映射(inverse mapping)(x, y) = A-1(x', y'):输出图像上像素位置(x',y')使用反向的坐标关系(仿射变换中,可以理解为变换矩阵A的逆)计算输入图像中的对应位置(x,y)。然后在最近的输入像素之间进行内插,计算的结果就是输出像素值。
虽然一般而言,对于仿射变换这样的特殊变换,其是一对一的映射关系。 但是实际中,由于图像的离散特性,在对输出结果的插值赋值的过程中,可能会有多个像素坐标映射到输出图像的同一位置,这就会产生如何将多个输出值合并为单个像素值的问题,另外也可能出现输出图像的某些位置完全没有相应的输入图像像素与它匹配,也就是没有被映射到。因此通常用反向映射来实现图像变换,从而将可能的“多对一”转变为“一对多”,进而避免了这一问题。就实现而言,反向映射更有效,因此也被众多软件所采用。