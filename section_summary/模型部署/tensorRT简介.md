## tensorRT安装

## 构建一个tensorRT引擎
1、构建一个网络定义
2、为构建器指定一个配置
3、调用构建器来创建引擎

NetworkDefinition接口（C++）被用来定义模型
可使用onnx或者C++编写
推荐使用onnx！！！
所以模型转换的步骤：
pytorch--->onnx--->tensorRT engine
注意事项：onnx转换只能是全有或者全无的，也就是但凡有一个模块tensorRT没有，就转换失败。
