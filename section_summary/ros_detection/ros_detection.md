# ROS Detection
终于，我对这个下手了，之前就一直想对这个进行一个实现。在研一的时候有一门课程简单的学了学ros，说实话，就是简单的跑跑代码，然后针对一些简单的地方改了改。也就是我们感知的算法，是需要进行ros封装，但是这部分没有了解过，但是我大体上是有一个了解的，所以这里来做一个简单的实现！让自己的知识体系更完整。
### 需求
接下来我们先说下本篇文章想要实现什么。首先我们都知道想到放到ros里，我们需要做的就是需要一个节点发布图片，如果是视频的话，就是发布每一帧的图片；然后需要一个节点订阅图片，然后实现图片的目标检测，并发布可视化的结果。这么看理论上只需要两个节点。这篇文章主要就是完成这个，首先我将利用python，也就是rospy做这件事情，因为两个节点，所以还需要一个需求就是launch文件，还有一个需求，就是开机自启。
在完成这些之后呢，我自己也不太清楚，能不能用cpp写这两个节点，然后运行py文件，这部分就需要进行探究。尽管我感觉这部分没有意义。因为底层还是py去运行inference，那么我可不可以用cpp写一个inference。这些都是后续拓展的需求。当然我肯定会满足最基本的需求。
### 开始
首先，放一个链接，也就是我在学习过程中参考的一个非常短的视频教程：【如何花10分钟把一个目标检测算法封装到ros中(以YoloX为例)】 https://www.bilibili.com/video/BV1LK411R7dw/?share_source=copy_web&vd_source=7af49fc7f4527a070bbd7df512e8c736
然后无所谓ros中封装的是什么算法，就是yolov5还是yolox都是可以的。代码是基于yolox代码改的，yolox代码参考：https://github.com/bubbliiiing/yolox-pytorch
因为是为了更好的学习，所以并没有直接使用视频中的代码，个人感觉跑一下也没啥用。所以这里准备自己简单改下代码，然后去做一下这个事情。okk接下来我们言归正传。因为一开始没有相机嘛，所以我们需要一个节点，来做图片的发布器。
### image_publisher
直接上代码，图片发布的节点
```ruby
#!/usr/bin/env python
```
每个 Python ROS 节点的顶部都有这个声明。第一行确保您的脚本作为 Python 脚本执行。
```ruby
import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from ros_numpy.image import numpy_to_image
```
这里的import前几个稍微有些了解，就不多解释啦，主要是后面几个包。rospy、sensor_msgs、还有ros_numpy。我们先说第一行注释的那条代码，这行代码是必须加的。别问我为什么知道，因为不加会报错。
接下来就是简单说一说import中的库。我打算从ros wiki入手，直接拿捏！
但是我在wiki去找，链接：http://docs.ros.org/en/melodic/api/rospy/html/
同时给大家推荐一个学习ros的网址：http://fishros.com/#/fish_home
这先只是针对常用的进行学习
```ruby
init_node(name, argv=None, anonymous=False, log_level=None, disable_rostime=False, disable_rosout=False, disable_signals=False, xmlrpc_port=0, tcpros_port=0)
#name：节点名字、anonymous：如果为 True，将使用 name 作为基础为节点自动生成一个名称。当您希望拥有同一节点的多个实例并且不关心它们的实际名称（例如工具、图形用户界面）时，这很有用。其余参数不关紧要。
is_shutdown()
#Returns: bool
sleep(duration)
#在 ROS 时间内休眠指定的持续时间。如果持续时间为负，则睡眠立即返回。
#Parameters:
#duration (float or Duration) - seconds (or rospy.Duration) to sleep
spin()
#阻塞直到 ROS 节点关闭。让出活动给其他线程。
#=======publisher类初始化========#
__init__ ( self , name , data_class , subscriber_listener = None , tcp_nodelay = False , latch = False , headers = None , queue_size = None )
name(str) - rostopic
data_class( Message class) - 传递消息的类别。
queue_size(int) - 用于异步发布来自不同线程的消息的队列大小。大小为零意味着无限队列，这可能很危险。当未使用关键字或传递 None 时，所有发布将同步发生，并打印一条警告消息。
#======publisher调用=======#
publish(self, *args, **kwds)
#向该主题发布消息数据对象。可以使用要发布的消息实例调用发布，也可以使用新消息实例的构造函数参数调用发布，即：
 pub.publish(message_instance)
 pub.publish(message_field_1, message_field_2...)            
 pub.publish(message_field_1='foo', message_field_2='bar')
Parameters:
args - Message instance, message arguments, or no args if keyword arguments are used
kwds - Message keyword arguments. If kwds are used, args must be unset

#=======rospy.rate初始化===========#
__init__(self, hz, reset=False)
hz (float) - 确定睡眠的赫兹率
reset (bool) - 如果为 True，则当 rostime 向后移动时重置计时器。[默认值：假]
#=======rospy.rate.sleep初始化===========#
sleep(self)
#尝试以指定的速率睡眠。sleep() 考虑自上次成功的 sleep() 以来经过的时间。

#=========rospy.Subscriber初始化========#
__init__ ( self , name , data_class , callback = None , callback_args = None , queue_size = None , buff_size = 65536 , tcp_nodelay = False )
# name(str) - rostopic
# data_class（消息类）- 用于消息的数据类型类，例如 std_msgs.msg.String
# callback(fn(msg, cb_args)) - 接收到数据时调用 ( fn(data)) 的函数。如果设置了 callback_args，函数必须接受 callback_args 作为第二个参数，即 fn(data, callback_args)。注意：可以使用 add_callback() 添加其他回调。
# callback_args(any) - 传递给回调的附加参数。当您希望为多个订阅重用相同的回调时，这很有用。
# queue_size(int) - 一次接收的最大消息数。这通常是 1 或无（无限，默认）。如果设置了此参数，则应增加 buff_size，因为传入数据在被丢弃之前仍需要位于传入缓冲区中。将 queue_size buff_size 设置为非默认值会影响该进程中该主题的所有订阅者。
```
rospy就先看这些，接下来我们了解下数据类型之类的,ros中有一个common_msgs,其中包含了很多种类，我们其中用到的就是传感器消息sensor_msgs中的Image。然后就是ros_numpy，这个库主要做的就是数据类型的转化，它的作用和cv_bridge的作用是基本没差的。代码中ros_numpy.image import的numpy_to_image和image_to_numpy就是将numpy与传感器sensor的image信息进行转换。
到这里就基本有一个大致的了解啦。
这个是一个照片的发布节点的写法。整体下来，就发现实际上这一块就是一个固定的范式。所以还是一个多写，熟练的问题。
```ruby
if __name__ == '__main__':
    image_dir = '/home/ktd/rpf_ws/yolox-pytorch/VOCdevkit/VOC2007/JPEGImages/'
    image_list = os.listdir(image_dir)
    image_list = [os.path.join(image_dir, i) for i in image_list]
    image_iter = iter(image_list)
    rospy.init_node('image_publish')
    publisher = rospy.Publisher('/test_image', Image, queue_size=1)
    loop_rate = rospy.Rate(1)
    while True:
        try:
            image_path = next(image_iter)
        except StopIteration:
            image_iter = iter(image_list)
            image_path = next(image_iter)
        image = cv2.imread(image_path)
        # import pdb;pdb.set_trace()
        image_msg = numpy_to_image(image, encoding='bgr8')
        publisher.publish(image_msg)
        loop_rate.sleep()
```
接下来我们就直接看yolox_ros那个节点，就可以啦。
```ruby
class YoloxRos(object):
    def __init__(self, predictor):
        self.predictor = predictor
        self.image_subscriber = rospy.Subscriber('/galaxy_camera/image_raw', Image, callback=self.image_callback, queue_size=1)
        self.image_publisher = rospy.Publisher('/image_publish', Image, queue_size=1)
    def image_callback(self, msg):
        image = image_to_numpy(msg)
        image = img.fromarray(image)
        r_image = self.predictor.detect_image(image)
        r_image = np.array(r_image)
        # result_image = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)
        self.image_publisher.publish(numpy_to_image(r_image, encoding='bgr8'))
if __name__ == "__main__":
    yolo = YOLO()
    rospy.init_node('yolox_ros')
    yolox_ros = YoloxRos(predictor=yolo)
    rospy.spin()
```
这里只放了主要的ros节点的函数，对于目标检测的函数，不再说明了。在主程序中，首先就是初始化一个yolox_ros的节点，然后第一行就是实例化YOLO类，这个类就是负责目标检测的类，在这个类中，可以直接使用yolo.detection函数进行目标检测。然后最后一行的rospy.spin()只是使节点无法退出，直到该节点已关闭。
然后就看下YoloRos这个类，初始化需要设定一个self.predictor，就是目标检测的函数，然后再初始化一个订阅信息的话题，然后初始化一个发布信息的话题。可以注意到在订阅信息的初始化中，多了一个callback，对没错，这里初始化之后，接收到的图片都会 先进行callback函数。还是很好理解的。
到这里基本就实现啦，easy~
同时，之后launch文件的书写，以及相机的驱动，都做了，可以看到上面代码的订阅话题不是图片发布话题的名字，是因为我改到了工业相机发布的话题。
最后放一个实现的图吧。
@import "rviz_vis.png"
这里就没必要在意所谓的检测精度啦哈哈哈哈。这里有一个细节，图像比较暗，我试了下，硬件上，应该是已经把光圈放到了最大，所以如果想要调整亮度，那就只能调一下曝光时间。
稍微调整了一下曝光时间和白平衡，是不是感觉好看些了呢，但是还有一个问题，确实解决不了这个反光问题，尤其是在照我的键盘的时候，白色的，反光真的看不清！
@import "camera.png"
