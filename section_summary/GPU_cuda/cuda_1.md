# 初识GPU
看了一篇文章，简单来说。GPU可以进行大量简单运算，但是缓存单元和控制单元较小。而CPU可以进行少量计算，但是可以计算复杂计算，同时拥有较大的缓存单元和控制单元。对比之后我们可以反观深度学习中的计算，卷积计算实际上就是简单的数学计算。所以利用GPU可以大量进行简单计算的特性，而且还是并行的。因此，gpu会比cpu快的多。同时这里，我好像也理解了一个问题，那就是：之前听别人说过，对于深度学习的模型的调用很简单，没有竞争力的。我们可以做一些更底层的东西增加自己的竞争力，比如说了解如何写一个cuda算子，这里我们可能就能更好的理解了，这里所说的cuda算子是不是就是用代码的形式写一个数学上的计算过程。然后将这个过程放在应用到gpu中，更方便计算。简单来说，我们写的cuda算子就是在教会gpu一个新的计算能力，同时这个计算方法我们在模型中也有使用，这样就可以加快我们模型运算的能力了。
GPU的两大特点：计算密集、数据并行
## CUDA
早期GPU的图像接口是使用Opengl和directx。后来NVIDIA公司就发布了cuda c语言。
GPU架构基本组成：
SP：最基本的处理单元，streaming processor 也称为CUDA core。最后具体的指令和任务都是在SP上处理的。GPU的并行计算也就是多个sp同时计算。
SM：多个sp加上其他的一些资源组成一个streaming multiprocessor。也叫做GPU大核，其他资源包括：warp scheduler，register，shared moemory等。SM可以看做是gpu的心脏，register，shared moemory是SM的稀缺资源。
thread：一个cuda的并行程序会被以许多个threads来执行。
block：数个threads会被群组成一个block，同一个block中的threads可以同步，也可以通过shared memory通信。
grid：多个block则会构成grid
warp：GPU执行程序时的调度单位，目前cuda的warp的大小为32，同在一个warp的线程，以不同数据资源执行相同的指令，这就是所谓的SIMT
是不是感觉很迷糊，确实是这样的。
我们往后看就好了。
## 安装
这里我们就省略了，网上的资料很多，只要能完成下面的显示基本就是成功了吧
```ruby
(base) ktd@ktd-Alienware:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Nov__3_21:07:56_CDT_2017
Cuda compilation tools, release 9.1, V9.1.85
```
## cuda模型理解
cuda软件体系
cuda提供了两层api来调用底层gpu硬件：cuda驱动api、cuda运行时api
cuda函数库：CUFFT：利用cuda进行傅里叶变换的函数库、CUBLAS：利用cuda进行加速的完整标准矩阵与向量的运算库、CUDPP：并行操作函数库、CUDNN：利用CUDA进行深度卷积神经网络
cuda应用程序：cuda程序分为两部分：主机host代码和设备device代码。设备代码会在编译时从主机分离，又GPU并行线程执行，主机代码由CPU执行。
一般代码的执行流程为：
分配host内存，并进行数据初始化（CPU初始化）
分配device内存，并从host将数据拷贝到device上（GPU初始化）
调用cuda的核函数在device上完成指定的运算（GPU并行运算）
将device上的运算结果拷贝到host上（将GPU结果传回CPU）
释放device和host上分配的内存（初始化清空）
线程模型：
cuda的线程模型丛小往大依次是：threads线程并行的基本单元--->block线程块---->grid网格，由一组block组成
block的特点：以1维2维或3维组织，允许彼此同步，可以通过共享内存快速交换数据
grid的特点：以1维、2维组织，共享全局内存
核函数！！！kernel
kernel是在device上线程中并行执行的函数，是软件概念，核函数用```__global__```符号声明，并用```<<<grid,block>>>```执行配置语法指定内核调用的cuda线程数，每个kernel的thread都有一个唯一的线程ID，可以通过内置变量在内核中访问。block一旦被分配好SM，该block就会一直驻留在该SM中，直到执行结束。一个SM可以同时拥有多个blocks
warp是SM的基本执行单元，也称线程束，一个warp有32个并行的thread，一个warp中所有的thread一次执行一条公共指令，并且每个thread会使用各自的data执行该指令。
线程索引：
这里有几个参数我们需要了解
```ruby
threadIdx.x#线程序号
blockIdx.x#线程块序号
gridDim.x#网格总序号数
index = blockIdx.x * blockDim.x + threadIdx.x#一维情况下，线程序号的计算方法
ix = threadIdx.x + blockIdx.x * blockDim.x#二维情况下，x方向线程序号的计算方法
iy = threadIdx.y + blockIdx.y * blockDim.y#二维情况下，y方向线程序号的计算方法
idx = iy * nx +ix
```
这一部分偏理论，我的建议是多看看大佬们的总结，我也是只看了《GPU高性能编程：cuda实战》这本书有了较为模糊的理解。
## CUDA C编程简介
这里我们先看一个通过调用cuda_runtime.h中的api得到gpu的一些属性。
注意：在编写cuda c程序时，需要将文件后缀命名为.cu，一般使用nvcc进行编译运行，支持c、c++语法
```ruby
#include<iostream>
#include<cuda.h>
#include<cuda_runtime.h>
int main() {
int dev = 0;
cudaDeviceProp devProp;
cudaGetDeviceProperties(&devProp, dev);
std::cout << "GPU Device Name" << dev << ": " << devProp.name << std::endl;
std::cout << "SM Count: " << devProp.multiProcessorCount << std::endl;
std::cout << "Shared Memory Size per Thread Block: " <<
devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
std::cout << "Threads per Thread Block: " << devProp.maxThreadsPerBlock <<
std::endl;
std::cout << "Threads per SM: " << devProp.maxThreadsPerMultiProcessor <<
std::endl;
std::cout << "Warps per SM: " << devProp.maxThreadsPerMultiProcessor / 32 <<
std::endl;
return 0;
}
#=========输出
GPU Device Name0: NVIDIA GeForce RTX 3080
SM Count: 68
Shared Memory Size per Thread Block: 48 KB
Threads per Thread Block: 1024
Threads per SM: 1536
Warps per SM: 48
```
通过运行代码，可以直到我们gpu型号，SM数量为68，每个线程的共享内存空间为48kb，每个线程块的线程数是1024，每个SM下的线程为1536，每个SM有48个线程束
##### 家人们，我破防了，为了学习这本书，我熬夜复习c语言两天，结果一看这文章用c++写的！！！！

## 内存管理
这里就是简单的api调用，虽然我在复习c的时候也没有把那几个函数搞明白（malloc，memcpy，memset，free）
这里在cuda使用的话，就是简单在这些函数前面加上cuda，但是要注意大小写(cudaMalloc,cudaMemcpy,cudaMemsset,cudaFree)
接下来简单介绍一下上面几个函数
```ruby
__host__ cudaError_t cudaMalloc(void** devPtr , size_t size)
#devPtr：开辟数据的首指针
#size：开辟的设备内存空间长度
```
这样看真的抽象，我们简单找个代码进行分析吧。这个代码是上面提到的那本书中的一个很简单的例子，就是在gpu上实现相加的操作。
```ruby
#include "../common/book.h"
__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}
int main( void ) {
    int c;
    int *dev_c;
    HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );
    add<<<1,1>>>( 2, 7, dev_c );
    HANDLE_ERROR( cudaMemcpy( &c, dev_c, sizeof(int),
                              cudaMemcpyDeviceToHost ) );
    printf( "2 + 7 = %d\n", c );
    HANDLE_ERROR( cudaFree( dev_c ) );

    return 0;
}
```
我想看这个代码就会比上面的清晰很多，代码通过使用```HANDLE_ERROR( cudaMalloc( (void**)&dev_c, sizeof(int) ) );```进行内存分配，其中&dev_c表示将指针变量的地址分配给gpu，分配的大小就是sizeof(int)。至于前面的```HANDLE_ERROR()```不太明白，根据书里说的，这个应该是book.h中声明的一个错误判断的东西吧。
接下来
```ruby
__host__ cudaError_t cudaMemcpy(void* dst , const void* src , size_t count , cudaMemcpyKind kind)
#dst 目的数据内存首指针
#src 源数据首指针
#count 数据长度
# kind 拷贝类型，cudaMemcpyDeviceToHost：从设备向主机拷贝
#cudaMemcpyHostToDevice：从主机向设备拷贝，cudaMemcpyHostToHost主机到主机，cudaMemcpyDeviceToDevice从设备到设备
```
我们继续把最简单的cudafree也介绍一下把
```ruby
__host__ cudaError_t cudaFree(void* devPtr)
#devPtr：设备变量指针
```
在例子中代码这两块的应用为
```ruby
cudaMemcpy( &c, dev_c, sizeof(int),   cudaMemcpyDeviceToHost ) 
cudaFree( dev_c )
```
因为之前已经在gpu上创建了内存空间，也就是说将dev_c这个指针变量已经存放在gpu了，接下来在gpu进行计算，然后将结果保存在dev_c所指向的int类型变量中，那么接下来就需要将这个变量从gpu复制到CPU中了，所以在cudaMemcpy的最后，设置的类型为cudaMemcpyDeviceToHost。至于前面的参数，sizeof(int)是拷贝内存大小，这个没的说。第一第二个参数分别表示拷贝过程中的目标和源头。也就是将dev_c这个指针变量拷贝到int类型变量c的地址中。
另外这里都提到了cudaError_t，这是以枚举的形式保存各种错误类型。
那本GPU高性能编程，确实太基础了这么一看，因为我看了前面的五章了，只是在教会我很基础的东西。但是这篇文章马上就来新东西了。也有可能c++让人开心的原因。具体的函数调用可以参考https://docs.nvidia.com/cuda/cuda-runtime-api/index.html
```ruby
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <math.h>
int main() {
float dets[6][4] = {
{23, 34, 56, 76},
{11, 23, 45, 45},
{12, 22, 47, 47},
{9, 45, 56, 65},
{20, 37, 55, 75},};
# copy data to gpu
std::cout << sizeof(dets) << std::endl;
float *dev_dets;
cudaError_t err = cudaSuccess;
err = cudaMalloc((void **)&dev_dets, sizeof(dets));
if (err != cudaSuccess) {
printf("cudaMalloc failed!");
return 1;
}
cudaMemcpy(dev_dets, dets, sizeof(dets), cudaMemcpyHostToDevice);
std::cout << "Copied data to GPU.\n";
# get back copied cuda data
float host_dets[sizeof(dets)/sizeof(float)];
cudaMemcpy(&host_dets, dev_dets, sizeof(dets), cudaMemcpyDeviceToHost);
std::cout << "Copied from cuda back to host.\n";
std::cout << "host_dets size: " << sizeof(host_dets) << std::endl;
for (int i=0;i<sizeof(dets)/sizeof(float);i++) {
std::cout << host_dets[i] << " ";
}
std::cout << std::endl;
cudaFree(dev_dets);
std::cout << "done.\n";
return 0;
}
//
输出为
96
Copied data to GPU.
Copied from cuda back to host.
host_dets size: 96
23 34 56 76 11 23 45 45 12 22 47 47 9 45 56 65 20 37 55 75 0 0 0 0
done.
```
我觉着根据我的c++实力，这样简单的代码应该没问题。首先定义一个二维数组，没啥问题二维数组所占内存大小=4 * 6 *4=96没啥问题。额，我简单的去官方文档看了看error handing这块的解释，我发现哈，看不懂哈哈哈哈。那就先跳过这部分吧。然后后面就很简单了。
上面代码是使用的cudaMalloc来申请设备内存，但是二维数组不推荐这么做，在kernel运算时较高的性能损失，CUDA给出了二维数组专用的内存申请函数cudaMallocPith，在设备间内存拷贝时，也要使用cudaMemcpy2D函数，参数如下
```ruby
__host__ cudaError_t cudaMallocPitch(void** devPtr , size_t* pitch , size_t width , size_t height)
#devPtr：开辟矩阵的数据的首指针
#pitch：分配存储器的宽度
#width：二维数组的列数
#height：二维数组的行数
__host__ cudaError_t cudaMemcpy2D(void* dst , size_t dpitch , const void* src , size_t width , size_t height ,  cudaMemcpyKind kind)
#dst 目的矩阵内存首指针
#dpitch ：dst指向的2D数组中的内存宽度，以字节为单位，是cuda为了读取方便，对其过的内存宽度，可能大于一行元素占据的实际内存
#src 源矩阵首指针
#spitch：src指向的2D数组中的内存宽度
#width src指向的2D数组中一行元素占据的实际宽度，为width * sizeof(type)
#height：2d数组的行数
# kind 拷贝类型，cudaMemcpyDeviceToHost：从设备向主机拷贝
#cudaMemcpyHostToDevice：从主机向设备拷贝，cudaMemcpyHostToHost主机到主机，cudaMemcpyDeviceToDevice从设备到设备
```
在了解这些之后，我想我们肯定很想将上面代码中的函数换掉，使其效率更高。所以我们需要怎么改呢？我们首先可能需要多几个变量，然后就是那两行改一下就可以了，那么下面就只写新的代码了。
```ruby
#因为在函数包含了新的参数，例如width和height等====但是它这里用的是size_t，这是一个什么数据类型呀
#size_t 是一些C/C++标准在stddef.h中定义的，size_t 类型表示C中任何对象所能达到的最大长度，它是无符号整数。
size_t width = 4;
size_t height = 6;
size_t pitch;
#提问，这里使用int定义行不行？
err = cudaMallocPitch((void **)&dev_dets, &pitch, sizeof(float)*width, height);
cudaMemcpy2D(dev_dets, pitch, dets, sizeof(float)*width, sizeof(float)*width,height,cudaMemcpyHostToDevice);
```
看了代码之后，不知道大家有没有疑问，这里的pitch为什么没有值。还有一点需要注意的就是，所有的宽度都是内存大小sizeof(float)*width，而行数就只是height，这个问题感觉和之前复习c语言时：定义二维数组，我们可以将行数不写，但是一定要写列数，实际上呢，就是代码需要知道行与行首地址之间到底需要多少多少内存空间。
这两个函数应该会使kernel的运行时间变短，因为pitch对齐后，可实现global内存联合访问，但是cudaMallocPitch和cudaMemcpy2D会变慢，因为比一维的操作多了对齐的考虑。
## kernel函数
想必一开始就会好奇```__global__```这种是干嘛用的。简单介绍下
```ruby
__device__ :在设备上执行，只能在设备上调用
__global__ : 在设备上执行，只能在主机上调用
__host__ : 在主机上执行，只能在主机上调用
```
有那么一点点了解了把，还有更深奥的，我不是很理解。
__device__和__global__代表函数在设备上执行，不支持递归，不能在函数体内声明静态变量，静态变量对应于CPU的整个程序生命过程，不能有可变长参数。__global__和__host__不能一起使用，而__device__和__host__可以一起使用，编译器会在cpu和gpu各复制一份函数。
不添加限定词，函数默认使用__host__,也就是在主机上执行
所有的kernel函数返回类型都是void，且kernel函数都是异步执行。
接下来就是kernel函数的调用了，想必大家在看了最开始那个例子的代码后，一定很好奇```add<<<1,1>>>( 2, 7, dev_c );```这是什么，没错，这就是对kernel函数的调用。首先我们看官方的解释
```ruby
kernel_func <<<Dg, Db, Ns, S>>> (param list);
#<<<Dg, Db, Ns, S>>>运算符内是核函数的执行参数，告诉编译器运行时如何启动核函数
#后面的param list实际上和我们c语言中调用函数一样，所以下面简单说下尖括号中的参数
#Dg：grid的维度和尺寸，dim3类型，意为一个grid有多少个block
#Db：block的维度和尺寸，dim3类型，意为一个block有多少个thread
#Ns（可选）：用于设置每个block除了静态分配的share memory以外，最多能动态分配的shared memory大小，单位为byte，不需要动态分配时，该值为0或者不写
#S（可选）：cudastream类型的参数，表示该核函数处在哪个流之中。
```
帮大家总结一下，可选的那两个，看不懂所以不写了，只要知道前两个就行了，一个是设置线程块的个数，一个是设置每个线程块中的线程数。所以在上面最简单的代码就可以看懂了。但是偏偏这篇文章搞了一个没见过的东西，因为在官方定义中，前两个参数实际上可以是3维度的一个数据，因此它定义了grid和block代码如下
```ruby
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
__global__ void printThreadIndex() {
int ix = threadIdx.x + blockIdx.x * blockDim.x;
int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy*blockDim.x * gridDim.x + ix;
if(threadIdx.x == 3 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y ==
1){
printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d, %d), global
index %2d \n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx);
}}
int main() {
dim3 grid(2, 3, 1), block(4, 2, 1);
cudaDeviceSynchronize();
printThreadIndex<<<grid, block>>>();
#// cudaDeviceSynchronize();
cudaDeviceReset();
return 0;
}
```
这里代码其实很简单，唯一不同的就是grid和block定义为了多维的情况，我印象中虽然这里是dim3的类型，但是实际上还是定义的二维度，我就印象中有这么个东西，下面是书里的原话
```#cuda运行时希望得到一个三维的dim3值，虽然当前不支持3维的线程格，但cuda运行时仍希望得到一个dim3类型的参数，只不过最后一维的大小为1。当仅用两个值来初始化dim3类型变量时，cuda运行时会将第3维的大小指定为1。```
所以我们这里就定义了线程块为(2,3)每个线程块的线程数为(4,2),如果你还不清楚，可以画出来，一目了然。所以在多个线程异步执行时，只有特定的线程，会打印输出，这里特殊的线程就是指threadIdx.x == 3 && threadIdx.y == 1 && blockIdx.x == 0 && blockIdx.y ==1
同时我们也可以手动计算ix，iy和idx看程序输出是否一致。
ix = threadIdx.x + blockIdx.x * blockDim.x = 3  + 0 * 4 = 3
int iy = threadIdx.y + blockIdx.y * blockDim.y = 1 + 1 * 2 = 3
unsigned int idx = iy*blockDim.x * gridDim.x + ix = 3 * 4 * 2 + 3 = 27
最终你会发现程序输出和你所计算的值是一致的，恭喜你，学会了代码还学会了数学。
## 就酱紫
怎么有人周六考试，还写这个呀，呜呜呜呜，我是真的喜欢。指针那块很绕，我也不是很清楚，如果有错那就有错吧，后面会改的。其实这只是一个开始，后面东西多着呢，太累了，休息了，最后！欢迎来到cuda的世界！