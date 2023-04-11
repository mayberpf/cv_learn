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