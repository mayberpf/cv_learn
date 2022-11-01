# 27
# TensorRT模型部署流程
1、使用pytorch或者其他训练框架训练模型并将模型保存下来。
2、模型导入tensorRT转化Wie引擎文件engine，并将engine文件序列化保存，这样后面可以快速调用它来执行模型的加速推理，无需再重新编译转换。
    以pytorch为例：
    首先需要将模型从pth格式转换为ONNX格式，然后再将ONNX的模型转换为TensorRT推理引擎
    ONNX全称叫做Open Neural Network Exchange ，它是用于表示深度学习模型的一种文件格式，可以作为中建桥梁，使模型能够在不同框架之间转换。
