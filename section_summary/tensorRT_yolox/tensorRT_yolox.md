# TensorRT yolox
许久不见，回到学校的第一篇文章，学校的生活我只能说是过于快乐了，也就是说基本上学习的时间并不多，所以我也没有寄很多的希望在学习一些新知识上，所以我做的第一件事就是做简历，毕竟马上就要秋招招工作了，目前简历基本已经做好了，但是问题是突然有一个新奇的想法，那就是我准备去投几个简历，然后就可以去实习！但是这样的话，我可能就回不去济南啦~
其次做的第二件事，就是tensorRT的学习吧，接下来我们就直接不如正题。
首先简单介绍：
推理inference加速，实际上就是将torch模型转换到可以使用trt加载的engine。不过其中有多种方法，1、可以先转onnx，然后再转换到trt engine；2、可以直接通过torch2trt进行转换；这里主要介绍的就是通过torch2trt进行转换。
同时我们要清楚一件事，那就是我们在转换engine时，一定要在对应显卡机器上操作。也就是我准备用车上的工控机进行inference，那么你就要在工控机上做转换，因为转换的过程，机器需要在显卡上不断的跑，从而确定如何分配内存和显存。
期初我我学习的是另外一个仓库，并不是torch2trt，而是：https://github.com/shouxieai/tensorRT_Pro
但是发现一些问题，多了一个步骤，所以需要更多时间去理解学习，本篇文章主要介绍torch2trt的方法。
# 安装
安装部分比较复杂。因为要安装的东西比较多：TensorRT、opencv、torch2trt等
这里不介绍opencv的安装过程，只要到官网下载包，然后编译安装，这里需要注意的就是安装时的g++和gcc版本。
tensorRT的安装
首先tensorRT的安装并不推荐deb的方法安装，因为我整半天没整明白，通过查了网上的资料，我发现网上都说：如果你的cuda是用deb安装的，那么你的tensorRT可以使用deb进行安装，否则不行。所以这里主要介绍通过包安装的方法，其实比较简单。
首先就是官网下载安装包，然后进行解压。
到python目录下
```ruby
cd python
python -m pip install tensorrt-7.2.3.4-cp38-none-linux_x86_64.whl
```
这里有一个点，需要注意一下，你进入到Python目录中，你会发现有很多个安装包，选择哪个取决于你的Python版本。举个例子：我的python是3.9的，那么我就要选择cp39的。
安装完成之后，将环境变量写入到bashrc文件中
```ruby
export LD_LIBRARY_PATH=TensorRT解压路径/lib:$LD_LIBRARY_PATH
```
当然cuda环境也要在bashrc中指定
```ruby
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:$LD_LIBRARY_PATH
export CUDA_INSTALL_DIR=/usr/local/cuda-10.2
export CUDNN_INSTALL_DIR=/usr/local/cuda-10.2
```
这里需要确定的cuda的版本。
最后source一下就可以啦。
除此之外还需要使用到pycuda，这个可以直接使用pip进行安装，但是安装时请指定版本2019.1，如果不指定的话，会安装最新的，有可能会报错。
安装完成对tensorRT进行测试
```ruby
(torch) rpf@rpf-ZERO:~/rpf_code/YOLOX$ python 
Python 3.9.12 (main, Apr  5 2022, 06:56:58) 
[GCC 7.5.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorrt
>>> print(tensorrt.__version__)
8.0.0.3
```
但是一开始我安装tensorRT的时候测试不是这样的，报了一个错，忘记是什么，但是出现了一个词cudnn，我猜测是我的cudnn的版本有问题不适配，所以我去查看cudnn的版本，结果。。。我没装，我自己的电脑上。
所以在装完cudnn之后，才能测试成功。
接下来安装opencv即可，然后是torch2trt，可以直接在github中搜索，然后Git
```ruby
git clone https://github.com/NVIDIA-AI-IOT/torch2trt
cd torch2trt
python setup.py install --user
```
安装过程也比较简单，只需要到目录下，然后执行命令就可以啦。安装完成，之后可以进到Python中看能不能import。
主要的一些都已经安装完毕了。接下来就可以执行啦。
# 代码实践
链接：https://github.com/Megvii-BaseDetection/YOLOX
本篇文章主要涉及2个，一个是将torch转为trt的文件，也就是trt.py文件，另外一个就是yolox的推理inference文件，inference是用c++写的。
操作过程很简单
首先我们要做的就是对yolox的inference文件进行编译。
首先就是要修改一下cmakelists.txt文件
```ruby
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# cudnn
#include_directories(/usr/local/cuda/cudnn/v8.0.4/include)
#link_directories(/usr/local/cuda/cudnn/v8.0.4/lib64)
#tensorrt
include_directories(/home/bai/TensorRT-7.2.3.4/include)
link_directories(/home/bai/TensorRT-7.2.3.4/lib)
```
这里的cuda的地址对应上，然后因为安装cudnn是将lib64和include文件夹将cuda的原始lib64和include进行替换。所以这里不需要指定cudnn的目录了。当然也需要指定tensorRT的安装目录，也就是解压的那个目录。
然后就可以进行编译了
```ruby
cd ~/YOLOX/demo/TensorRT/cpp/
cmake .
make
```
编译结束之后，你就能看到一个可执行文件yolox
接下来就是运行trt.py文件，将torch的模型转换为engine
```ruby
python tools/trt.py -n yolox-s -c weights/yolox_s.pth.tar
```
恭喜，到这里马上就可以执行YOLOX进行检测
```ruby
./demo/TensorRT/cpp/yolox YOLOX_outputs/yolox_s model_trt.engine -i assets/dog.jpg
```
# 代码详解
我们都知道c++是一个很难的事情，最起码对我来说是这样的，所以先看下trt.py文件
本来是不想debug的，但是真的一调试就通，没有什么代码是看不懂滴
```ruby
def make_parser():
    parser = argparse.ArgumentParser("YOLOX ncnn deploy")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument(
        "-w", '--workspace', type=int, default=32, help='max workspace size in detect'
    )
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    return parser
```
首先就是加载一些参数的函数，这个函数中其实没什么很重要的参数，主要是--name 需要指定加载模型的名字，其次需要指定torch模型的权重文件地址。
在main函数中，首先加载上面定义的参数，其次根据上面定义的模型名字，加载对应的模型一些超参数。
```ruby
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
```
这些超参其实用处不大，主要包括输出地址、分类类别数、模型深度宽度、输入大小、学习率、数据增强等。
之后利用get_model()函数获取模型的结构。
```ruby
    if not args.experiment_name:
        args.experiment_name = exp.exp_name
    model = exp.get_model()
```
到这里就将模型的结构加载进来，说实话，没必要仔细看这块的，毕竟像这种官方的代码，都有一个问题，那就是代码写的难以理解，说白了，你要是直接加载模型，就直接用一个类做实例化不就好了，它这里就写的挺复杂，一层函数套一层函数。
接下来设定一个输出文件夹地址
```ruby
    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)
```
接下来要做的肯定就是模型权重的加载，利用cpu加载滴，其他没啥说的，就是很常规的事情。
```ruby
    if args.ckpt is None:
        ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    else:
        ckpt_file = args.ckpt
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
```
接下来将模型给到验证，也就是inference，然后模型给到cuda。最后一句说的decode_in_inference这里不是很理解，我找了找了，这里是在初始化head的时候设置的。作者给出的注释是：对于部署，设置为 False。所以这里设置了false。也就是说，肯定是在训练的过程需要一个这样的解码过程，但是在验证中就不需要。
```ruby
    logger.info("loaded checkpoint done.")
    model.eval()
    model.cuda()
    model.head.decode_in_inference = False
```
接下来先给出一个输入
```ruby
    x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
```
接下来就是文章中最最重要的，只需要利用这样的函数，就能够转换。我们最后对这个api进行解释。
```ruby
model_trt = torch2trt(
    model,
    [x],
    fp16_mode=True,
    log_level=trt.Logger.INFO,
    max_workspace_size=(1 << args.workspace),
    max_batch_size=args.batch,
)
```
然后捏，就是存储！我不是很理解，为什么还要存储engine里的参数到pth文件中，我感觉是没有什么意义的呀，因为只是需要一个engine文件就行了呀。然后把东西存在model_trt.engine中。
```ruby
    torch.save(model_trt.state_dict(), os.path.join(file_name, "model_trt.pth"))
    logger.info("Converted TensorRT model done.")
    engine_file = os.path.join(file_name, "model_trt.engine")
    engine_file_demo = os.path.join("demo", "TensorRT", "cpp", "model_trt.engine")
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())
```
最后做一个文件的复制
```ruby
shutil.copyfile(engine_file, engine_file_demo)
logger.info("Converted TensorRT model engine file is saved for C++ inference.")
# shutil.copyfile(file1,file2)
```
接下来简单看下torch2trt的api！这里没有在网上找到一些资料，所以我们进入到原函数中进行查看。可以对比上面的函数的使用，首先第一个参数就是model，这个是必须要有的，其次就是输入。剩下的就是一些高级设定啦，比如fp16、int8、max_batch_size等。
```ruby
def torch2trt(module,
              inputs,
              input_names=None,
              output_names=None,
              log_level=trt.Logger.ERROR,
              fp16_mode=False,
              max_workspace_size=1<<25,
              strict_type_constraints=False,
              keep_network=True,
              int8_mode=False,
              int8_calib_dataset=None,
              int8_calib_algorithm=DEFAULT_CALIBRATION_ALGORITHM,
              use_onnx=False,
              default_device_type=trt.DeviceType.GPU,
              dla_core=0,
              gpu_fallback=True,
              device_types={},
              min_shapes=None,
              max_shapes=None,
              opt_shapes=None,
              onnx_opset=None,
              max_batch_size=None,
              avg_timing_iterations=None,
              **kwargs):
```
okk，接下来我们看下c++的推理代码，非常滴难。只能说其实我只会一点皮毛，希望学完之后
```ruby
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "logging.h"
```
#include <fstream> 是C++程序中常用的预处理指令，它包含了fstream库。这个库提供了用于处理文件输入/输出的类。fstream库主要包括以下几个类：
std::ifstream：用于从文件读取数据的输入文件流类。
std::ofstream：用于向文件写入数据的输出文件流类。
std::fstream：用于同时进行文件的读取和写入操作的输入/输出文件流类。
#include <iostream>用于存储iostream类库的 源文件 ，在这个程序中用于提供输出这项功能。
<sstream>库定义了三种类：istringstream、ostringstream和stringstream，分别用来进行流的输入、输出和输入输出操作。另外，每个类都有一个对应的宽字符集版本。注意，<sstream>使用string对象来代替字符数组。这样可以避免缓冲区溢出的危险。而且，传入参数和目标对象的类型被自动推导出来，即使使用了不正确的格式化符也没有危险。
C++的<numeric>头文件中包含了一系列可用于操作数值序列（sequences of numeric value）的函数，通过修改函数的参数类型也可以将这些函数应用到非数值序列中。熟练使用<numeric>中的函数可以方便地对例如vector<>,list<>等容器进行数值计算。
std::chrono是在C++11中引入的，是一个模板库，用来处理时间和日期的Time library。要使用chrono库，需要include<chrono>，其所有实现均在chrono命名空间下。
```cpp
#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define DEVICE 0  // GPU id
#define NMS_THRESH 0.45 //NMS非极大值抑制阈值
#define BBOX_CONF_THRESH 0.3 //bboxes置信度阈值
```
Nvidia官方提供的TRT插件的API
```cpp
using namespace nvinfer1;
```
接下来定义一些有关输入输出的定义参数
```cpp
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 80;
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";
static Logger gLogger;
```
若一个变量声明为static,则该变量就会储存在程序的静态区域；那么在程序开始运行的时候，该变量就会一直存在，而不是随着所在函数运行终止而终止。（储存在栈中，该为类所有）
另外:static声明的变量必须在类外初始化；
若一个变量声明为const，则意味着该变量为不可变。所在类实例对象时，才会创建该变量（存在在堆中,该变量为对象所有）。
另外：const声明变量需要在初始化列表中初始化。
static const声明的变量必须在初始化时赋值。
const char*的类型是：“指向一个具有const限定符的char类型的指针”。（不能修改其值）
char*的类型是：“指向一个char类型的指针”
后面的代码首先给人一个感觉就是这一定是一个函数。那我们先看下这个c++函数的一个格式
```ruby
C++ 中的函数定义的一般形式如下：
return_type function_name( parameter list )
{
   函数体
}
```
对应到这个代码，就很容易理解,首先是数据的返回类型是cv下的一个mat，其次函数名叫做static_resize，这个输入只要传图像参数进去就行，函数会自己取地址，然后后面的操作好像不是很难懂，就是一个长短边的计算，然后将最长边转换到640，构造一个re的cv::mat，3通道图片，接下来用resize函数操作，和Python不一样的就是，Python不需要指定输出，而cpp需要指定输出。接下来构造一个out的mat，然后缩放后的re复制到out下的一个固定区域。
```cpp
cv::Mat static_resize(cv::Mat& img) {
    float r = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
    // r = std::min(r, 1.0f);
    int unpad_w = r * img.cols;
    int unpad_h = r * img.rows;
    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(img, re, re.size());
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(0, 0, re.cols, re.rows)));
    return out;
}
```
看到这里说实话，对c++越来月熟悉了，接下来我们就直接看main函数，然后一步一步往下走。首先main函数开头，这里只要明白，后面有一段就可以直接跳过理解，因为我感觉没有必要。
argc表示程序运行时发送给main函数的命令行参数的个数（包括可执行程序以及传参）。
argv[]是字符指针数组，它的每个元素都是字符指针，指向命令行中每个参数的第一个字符。
argv[0]指向可执行程序。
argv[1]指向可执行程序后的第一个字符串。
argv[2]指向可执行程序后的第二个字符串 。
argv[3]指向可执行程序后的第三个字符串 。
argv[argc]为NULL。
```cpp
int main(int argc, char** argv) 
```
了解了这个，再知道我们运行可执行文件的命令行，在后面的程序中会提到argv[1]、argv[2]、argv[3]以及agrc就知道是什么了。
```ruby
./demo/TensorRT/cpp/yolox YOLOX_outputs/yolox_s/model_trt.engine -i assets/dog.jpg
```
所以后面会有一段函数，实际上就是检测输入命令行的参数够不够，以及参数中的一些文件地址对不对。
```cpp
    if (argc == 4 && std::string(argv[2]) == "-i") {
        const std::string engine_file_path {argv[1]};
        std::ifstream file(engine_file_path, std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "run 'python3 yolox/deploy/trt.py -n yolox-{tiny, s, m, l, x}' to serialize model first!" << std::endl;
        std::cerr << "Then use the following command:" << std::endl;
        std::cerr << "./yolox ../model_trt.engine -i ../../../assets/dog.jpg  // deserialize file and run inference" << std::endl;
        return -1;
    }
    const std::string input_image_path {argv[3]};
//上面的就是检查命令行的参数够不够，以及engine的地址对不对。
//下面这一块说真的不是很懂。
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr); 
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    auto out_dims = engine->getBindingDimensions(1);
    auto output_size = 1;
    for(int j=0;j<out_dims.nbDims;j++) {
        output_size *= out_dims.d[j];
    }
    static float* prob = new float[output_size];
```
后面的代码就是读取图片+我们上面看的resize操作。
```cpp
    cv::Mat img = cv::imread(input_image_path);
    int img_w = img.cols;
    int img_h = img.rows;
    cv::Mat pr_img = static_resize(img);
    std::cout << "blob image" << std::endl;
```
blobfromimage这个函数仔细看源码不是很懂，可能需要debug去调试，根据网上的资料，这个函数主要是对图像进行预处理，以便在深度学习或图像分类中使用，主要进行的操作就是对像素值进行减均值以及像素值缩放等操作。
```cpp
    float* blob;
    blob = blobFromImage(pr_img);
    float scale = std::min(INPUT_W / (img.cols*1.0), INPUT_H / (img.rows*1.0));
```
有一点是可以明白的，那就是这个函数的返回值是一个float* 也就是指针。所以一开始定义了一个blob的指针。然后计算了一下最长边缩放的比例。
根据代码中的注释，接下来就要运行推理进行操作了，主要就是下面的四行代码。
```cpp
auto start = std::chrono::system_clock::now();
doInference(*context, blob, prob, output_size, pr_img.size());
auto end = std::chrono::system_clock::now();
std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
```
一共就只有四行代码，推理也就一行代码。剩余的三行代码都是用来对运行时间进行计时并打印的操作。记下来主要看下推理的代码。但是我发现，并看不懂。
查了下资料，怎么说，这个推理函数难道是一个定式？因为其实我们已经可以看到里面确实有很多断言函数。所以先简单了解下。
```cpp
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
```
这个用来明确输入输出，这个好像只是确定了一个名字。后面首先会进行分配GPU的内存，将输入从主机拷贝至GPU，然后执行推断。
```cpp
context.enqueue(1, buffers, stream, nullptr);
```
再往后具体就很难了，不是很了解。


