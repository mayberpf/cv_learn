# 第一讲
```ruby
mkdir -p test/src#-p 创建工作空间
catkin_make
catkin_create_pkg hello roscpp rospy std_msgs
touch hello.cpp
#编程要自己学哦！！！
source ./devel/setup.bash
rosrun hello hello_c
#python 在scripts文件夹下
ll#查看文件权限
#drwxrwxr-x 1 位是文件类型 2-4位是owner的权限 5-7位是group的权限 8-10位是other的权限 d代表directory(目录) rwx分别是read(读),write(写),execute(执行) - 在第一位指的是文件类型为文件，在后面的位上指的是不具有对应权限
chmod +x hello_p.py
chmod 777 /home/ubuntu/myblog
#鱼香ros--->ros安装
```
# 第二讲
launch文件的内容
```ruby
<launch>
    <node pkg = 'helloworld' type = "demo_hello" name = "hello" output = "screen">
    <node pkg = "功能包的名字" type = <节点名字> name = "为节点命名" output = "设置日志的输出目标">
<launch>
#运行launch文件
roslaunch 包名 launch文件名
#launch文件夹一般在功能包文件夹下面
```
# 第三讲
话题通讯
1、创建发布者
```ruby
#include <sstream>
#include "ros/ros.h"
#include "std_msgs/String.h"
int main(int argc, char **argv)
{
  // ROS节点初始化
  ros::init(argc, argv, "talker");
  // 创建节点句柄
  ros::NodeHandle n;
  // 创建一个Publisher，发布名为chatter的topic，消息类型为std_msgs::String
  ros::Publisher chatter_pub = n.advertise<std_msgs::String>("chatter", 1000);
  // 设置循环的频率
  ros::Rate loop_rate(10);
  int count = 0;
  while (ros::ok())
  {
	// 初始化std_msgs::String类型的消息
    std_msgs::String msg;
    std::stringstream ss;
    ss << "hello world " << count;
    msg.data = ss.str();
	// 发布消息
    ROS_INFO("%s", msg.data.c_str());
    chatter_pub.publish(msg);
	// 循环等待回调函数
    ros::spinOnce();
	// 按照循环频率延时
    loop_rate.sleep();
    ++count;
  }
  return 0;
}
```
2、创建订阅者
```ruby
#include "ros/ros.h"
#include "std_msgs/String.h"

// 接收到订阅的消息后，会进入消息回调函数
void chatterCallback(const std_msgs::String::ConstPtr& msg)
{
  // 将接收到的消息打印出来
  ROS_INFO("I heard: [%s]", msg->data.c_str());
}

int main(int argc, char **argv)
{
  // 初始化ROS节点
  ros::init(argc, argv, "listener");

  // 创建节点句柄
  ros::NodeHandle n;

  // 创建一个Subscriber，订阅名为chatter的topic，注册回调函数chatterCallback
  ros::Subscriber sub = n.subscribe("chatter", 1000, chatterCallback);

  // 循环等待回调函数
  ros::spin();

  return 0;
}
```
```ruby
rqt_graph#查看拓扑结构
```