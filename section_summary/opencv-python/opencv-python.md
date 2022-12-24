# opencv-python
最近在做一些传统图像处理的事情，找了很多算法，之后我发现实际上这些算法，很多opencv都有，我们甚至可以直接调用即可，也就是说好像我们需要对opencv里面的库有所了解，最起码就是我们得知道opencv库里包含的函数都能实现什么功能。
因此，本篇文章主要针对opencv的文档，做的一些笔记。主要就是简单介绍一下文档中个的各个函数。
# 函数调用
## 读取图像
```ruby
#为了后面好写，这里使用import
import cv2 as cv
import numpy as np
```
接下来我们就是简单的看下各种函数及其作用
```ruby
#读取图像
cv.imread(地址,读取方式)
#显示图像
cv.imshow(窗口名字,去读对象)
#键盘绑定函数---
cv.waitKey(0)
#破坏我们创建的窗口
cv.destroyAllWindows()
#破坏指定窗口
cv.destroyWindows()
#新建空窗口
cv.nameWindow()
cv.imwrite(文件名,保存对象)
```
提示：opencv加载的彩色图像处于BGR模式。但是matplotlib以RGB模式显示。
## 视频
```ruby
#读取摄像头视频
cv.VideoCapture(0)
#读取视频
cv.VideoCalture('视频名字')
```
## 绘图功能
```ruby
#画线
cv.line(图像,点坐标，点坐标，颜色，厚度)
#画矩形
cv.rectangle(图像，矩形左上角，矩形右下角，颜色，厚度)
#画圆圈
cv.circle(图像，中心坐标，半径，颜色，厚度)
# 椭圆
cv.ellipse()
#画多边形
cv.polylines(图像，[多个点])
#写字
cv.putText()
```
## 鼠标做画笔
```ruby
cv.setMouseCallback()
```
## 图像基本操作
opencv读取图片之后，会将图片转换为numpy的形式。
```ruby
img = cv2.imread()
#图像的形状
img.shape
#图像像素总数
img.size
#图像数据类型
img.dtype
#拆分图像通道
b,g,r = cv.split(img)#比较耗时的操作
#合并通道
img = cv.merge((b,g,r))
#创建相框
cv.copyMakeBorder()
```
## 图像的算法运算
```ruby
#图像加法，可以使用opencv或者numpy加法都是可以的
#opencv加法是饱和计算，numpy加法是模运算
cv.add()
#图像融合，对图像赋予不同权重，具备融合或者透明的感觉
cv.addWeighted()
#图像的位运算
```
## 改变颜色空间
```ruby
cv.cvtColor(图片,转换类型)
```
这里具体建议了解一下hsv和rgb
## 图像的几何变换
这里主要是图像的旋转，缩放和平移等。但是好像关于图像的旋转很少有人用opencv做的，不知道为什么，很多人都是使用PIL做的呀。可能使用PIL比较简单。
```ruby
#旋转
img = PIL.Image.open('图片路径')
img1 = img.rotate()
#缩放
cv.resize(对象，(resize后的大小)，插值方式)
#平移，opencv的旋转平移都是利用旋转平移矩阵做的
cv.warpAffine(对象,矩阵,(列数,行数))
#获取旋转矩阵
cv.getRotationMatrix2D((旋转中心点坐标),旋转角度)
#放射表换
M = cv.getAffineTransform(pts1,pts2)
dst = cv.warpAffine(img,M,(cols,rows))
#透视变换
M = cv.getPerspectiveTransform(pts1,pts2)
dst = cv.warpPerspective(img,M,(300,300))
```
## 图像阈值
这一部分就是指，像素值小于阈值的设置为0,大于则设置为最大值。
```ruby
#127是阈值，255是大于阈值的最大值
ret,thresh1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
#自适应阈值
cv.adaptiveThreshold
#Otsu的二值化
```
## 图像平滑
使用各种滤波，低通滤波器LPF有助于消除噪声，使图像更模糊。高通滤波HPF有助于在图像中找到边缘。
```ruby
#2D卷积图像过滤
cv.filter2D()
cv.blur()
cv.boxFilter()
#高斯模糊
cv.GaussianBlur()
#创建高斯内核
cv.getGaussianKernel()
#中位模糊---对于消除图像的椒盐噪声很有效
cv.medianBlur()
#双边滤波---去除噪声的同时保持边缘清晰
cv.bilateralFilter()
```
## 形态学转换
侵蚀---可以理解为变细？
```ruby
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
```
扩张---可以理解为变粗？
```ruby
dilation = cv.dilate(img,kernel,iterations = 1) 
```
开运算==侵蚀后扩张---消除噪音很有用
```ruby
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel) 
```
闭运算==扩张后侵蚀---在关闭前景对象内部的小孔或对象上的小黑点时很有用。
```ruby
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel) 
```
形态学梯度
```ruby
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel) 
```
顶帽==输入图像和图像开运算之差
```ruby
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel) 
```
黑帽==输入图像和图像闭运算之差
```ruby
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel) 
```
## 图像边缘检测
```ruby
edges = cv.Canny(img,100,200)
```
## 直方图
```ruby
#直方图计算
cv.calcHist（images，channels，mask，histSize，ranges [，hist [，accumulate]]）
hist,bins = np.histogram(img.ravel(),256,[0,256])
plt.hist(img.ravel(),256,[0,256])
```
## 霍夫变换
在霍夫变换之前，请进行canny边缘检测，才会有更好的效果。
```ruby
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,200)
```
## 图像分割

```ruby
#分水岭算法
markers = cv.watershed(img,markers) 
#连通区域
ret, markers = cv.connectedComponents(sure_fg)
```