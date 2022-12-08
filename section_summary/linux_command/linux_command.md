# Linux系统常用命令行
最近不是在准备汽车工程学的考试嘛，没有什么经历去学一些东西，做一些课题相关的东西。但是这个汽车工程学真的学不会呀，所以我准备做一点并不需要动脑子就能掌握的东西。经过这一学期的锻炼吧，我基本上已经对很多linux的命令较为熟悉了，所以想做一个总结。
这篇文章呢，主要介绍linux的一些常用的命令行，还有vim编译器的简单使用。
## linux_command
主要是参考linux命令行大全那本书，但是我会跳过一些不太常用的。下面这些是一试就懂的。
```ruby
date
#输出日期时间，相信我，你会觉着有用的
df
#查看磁盘驱动当前可用空间
exit
#关闭终端，但是我一般使用control+d
pwd
#查看当前目录
ls
ls -l
#列出当前目录的内容====这个真的常用
clear
#清楚当前终端的所有内容
```
简单看下cd，实际上常用的就几个，要么cd到下一个目录，要么cd到上一级，说白了很少有那种跨多级文件夹的情况
```ruby
cd username
#到下一级文件夹中
cd ..
#到上一级文件夹中
cd -
#到先前的工作目录中
cd 
#到主目录
```
看到了一个less命令，可以查看文件的内容，实际上我并不是常用这个，在实际应用中，我可能会选择使用cat指令，或者用vim去看，或者gedit也可以
```ruby
cat file_name
#cat会将文件的内容以打印的形式打印出来
vim file_name
#vim编辑器，使用了之后才知道，真的香
gedit file_name
#这个简直不要更简单
```
在介绍cp复制命令之前呢，先来说说mkdir这个超级棒的命令
```ruby
mkdir dir1
#创建一个叫做dir1的文件夹，使用这种方式也可以创建多个目录
mkdir -p dir1/dir2
#创建多级目录
```
接下来就是cp复制命令，这个感觉我们只是会将一个文件复制到其他文件夹，或者将文件夹复制到其他的文件夹
```ruby
cp item dir
cp item1 item2 dir
#将文件复制到文件夹中，其中这个文件其实可以是多个
cp dir1/* dir2
#这个是将dir1中的所有文件复制到dir2中，但是不能复制文件夹
cp -r dir1 dir2
#这个将dir1中的东西全部复制过去
```
移除和重命名文件，这个命令我基本不用。移动文件就相当于剪切
```ruby
mv item dir1
mv item1 item2 dir1
#这里的移动文件和上面的cp如出一辙
#包括后面的基本也一致，就不介绍了
```
接下来就是rm删除命令，真的不要太爽
```ruby
rm-rf /*
#为了更好的学习，建议大家实践一下。(实践完之后不要打我就好了)
#想要知道为什么，就去百度一下吧
rm item1 item2
#删除文件，可以删除多个文件
rm -d dir
#删除一个空文件夹
rm -r dir 
#删除文件夹
```
到此为止，很多常用的命令都已经说完了，但是还剩一个不一定是很常用的，但是一定是最重要的指令，那就是help，就是有时候可能你的命令行没有做到你想做的，这个时候你就要考虑是不是你的命令行有问题，所以你可以通过使用help来更好的理解这些命令，举个例子对于cp
我试了一下，所有的help的都挺长的，所以这里我放一个相对较短的
```ruby
(torch) ktd@ktd-Alienware:~$ rm --help 
用法：rm [选项]... [文件]...
Remove (unlink) the FILE(s).

  -f, --force           ignore nonexistent files and arguments, never prompt
  -i                    prompt before every removal
  -I                    prompt once before removing more than three files, or
                          when removing recursively; less intrusive than -i,
                          while still giving protection against most mistakes
      --interactive[=WHEN]  prompt according to WHEN: never, once (-I), or
                          always (-i); without WHEN, prompt always
      --one-file-system		递归删除一个层级时，跳过所有不符合命令行参
				数的文件系统上的文件
      --no-preserve-root  do not treat '/' specially
      --preserve-root   do not remove '/' (default)
  -r, -R, --recursive   remove directories and their contents recursively
  -d, --dir             remove empty directories
  -v, --verbose         explain what is being done
      --help		显示此帮助信息并退出
      --version		显示版本信息并退出

默认时，rm 不会删除目录。使用--recursive(-r 或-R)选项可删除每个给定
的目录，以及其下所有的内容。

To remove a file whose name starts with a '-', for example '-foo',
use one of these commands:
  rm -- -foo

  rm ./-foo

请注意，如果使用rm 来删除文件，通常仍可以将该文件恢复原状。如果想保证
该文件的内容无法还原，请考虑使用shred。

GNU coreutils online help: <http://www.gnu.org/software/coreutils/>
请向<http://translationproject.org/team/zh_CN.html> 报告rm 的翻译错误
Full documentation at: <http://www.gnu.org/software/coreutils/rm>
or available locally via: info '(coreutils) rm invocation'

```
你以为到这里就完了，其实并不是，我相信在后面的学习过程中，我还会接收到更多的命令行，并且熟悉他们。比如像echo打印一行文本、history显示历史命令、sudo获得管理员权限、chmod改变文件的权限、top查看当前所有任务的资源占用情况、htop查看cpu占用情况、kill杀死某个进程、shutdown关机、reboot重启、find查找文件（这玩意一直不懂），所以我也用locate、ifconfig等命令还有很多。

差点忘记了十分重要的vim，（vim是vi的升级版，反正我是将他两相等了）在这本书上的原话是这样的：vi用户界面的不友好是非常出名的，但是看一位vi专家在键盘前坐下并开始“演奏”，将是莫大的艺术享受。确实，在我研一刚入学的时候，我就领悟到了vim的不友好，到现在开始慢慢适应它，以及我很享受双手敲击键盘“演奏的过程”。
其实想要了解vim，我更建议去看一下大佬们的讲解，我这里只是简单讲解一下非常简单的操作，以至于你在遇到vim的时候不会手忙脚乱。
```ruby
vim item
#使用vim打卡一个文件，也可以是新建一个文件
```
在vim的使用，首先要清楚实际上它有很多模式，比如正常模式，插入模式，名字可能不正确但是意思差不多，还有什么，我忘记了。但是在使用的时候，我们无非就是阅读，或者码字修改。正常模式就是刚打开文件的模式。
最简单的使用，vim打开文件之后，我们摁```i```进入插入模式，这个时候你就可以随意的打字了。当你写完了呢，使用esc键即可退出该模式，然后使用```:wq```进行保存。这样我们就完成了一次最基本的修改。
值得注意的地方，就是这里的不管是i还是:wq都必须是英文才可以。中文是打不上的哦，所以当你输入了命令，但是左下角没有显示的话，不妨看看自己输入的是不是英文。
除此之外呢，当然还有很多命令，毕竟vim用的好，你就可以完全不需要鼠标了！！！
再介绍几个常用的把
```ruby
:q
#退出
:q!
#强制退出
```
在正常模式的时候，我们可以使用H,J,K,L来移动光标，但是我更喜欢使用上下左右。通过使用dd来删除当前行，或者5dd删除当前行和下面四行。通过使用yy来复制当前行，5yy复制当前行和后四行。那么我们就可以使用p来进行粘贴。除此之外呢，还要一个查找命令很常用，那就是```/要查找的内容
```ruby
/good
```
