# 空洞卷积
最近想到这个的原因是因为,希望减少计算量,并且增加感受野,同时联系上下文信息(印象中空洞卷积是可以做的)。接下来我们就简单的了解一下空洞卷积。实际上空洞的效果就是在普通卷积中插入一些参数为0的卷积核，插入多少的参数被定义为扩张率。dilation rate 
注意，这里的dilation =1表示的是普通卷积。
池化虽然能够增加感受野，但是池化是无法学习的，就是它是没有参数的，所以难免就会丢失一些信息，导致小物体的无法重建。使用空洞卷积，可以在不使用池化情况下且计算量相当，且提供更大的感受野。空洞卷积在卷积核中插入权重为0的值，因此每次卷积中会跳过一些像素点；并且空洞卷积得到的结果中，邻近像素是通过卷积上一层相互独立的特征点得到的，所以，结果的邻近像素之间缺少相关性。
（这里的缺少相关性，是不是可以理解为缺少上下文信息？？？）
以及还会造成远距离获取的信息没有相关性，因为空洞卷积稀疏的采样输入信号，使得远距离卷积得到的信息之间没有相关性。影响分类结果。而且因为增大了分辨率，导致在实际中不好优化，模型运行时速度会变慢。
看到这里，就感觉，这个空洞卷积确实不太适合我现在做的工作，最起码我最近替换的卷积，确实是不太合适的，并不可以使用空洞卷积去替代下采样的卷积 ，那么问题来了，能不能使用空洞卷积去替代模型中只改变通道数的普通卷积，而且在替换的过程中，是不是应该先将特征图进行升维，做完空洞卷积之后再进行降维。因为通道数增加了，就可以弥补特征图过滤的问题。我觉着这个我后面还可以试一试。
#### PS：上面说到的，使用空洞卷积替代下采样卷积，效果极差，直接导致我目标检测的正确率为0.0。这是为什么，我确实也不清楚。
但是空洞卷积确实可以扩大感受野，且降低计算量，一般都要进行降采样操作。传统来说，会使用池化或者步长为2的卷积，这样虽然可以增加感受野，但空间分辨率降低了，为了能不丢失分辨率，又想扩大感受野，可以使用空洞卷积。这在检测，分割中十分有用。因为一方面感受野打了可以检测分割大目标，另一方面分辨率高了可以精确定位目标。
此外通过空洞卷积来捕获多尺度上下文信息。在上述中，空洞卷积有一个参数可以设置为dilation rate ，具体含义就是在卷积核中填充dilation-1个0,因此，当设置dilation rate时，感受野就会不一样，即可以获取了多尺度信息。