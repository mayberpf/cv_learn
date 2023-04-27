# # 使用PIL.Image读取图片
# from re import S
# import torch
# from PIL import Image
# import pdb
# import torchvision.transforms as transforms
# def image_proprecess(img_path):
#     img = Image.open(img_path)
#     data_transforms = transforms.Compose([
#         # transforms.Resize((384, 384), interpolation=3),
#         transforms.ToTensor(),
#         # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#     data = data_transforms(img)
#     data = torch.unsqueeze(data,0)
#     return img, data

# image_path = '/home/ktd/rpf_ws/cv_learn/img/auto.png'
# # pdb.set_trace()
# img,data = image_proprecess(image_path)
# print(img)
# print(data)

# inputs = torch.rand((3,640,640))
# print(inputs.shape)
# conv = torch.nn.Conv2d(3,64,3,2,1)
# outputs = conv(inputs)
# print(outputs.shape)

import torch
import torch.nn as nn
from torch import autograd
# kernel_size的第哥一维度的值是每次处理的图像帧数，后面是卷积核的大小
m = nn.Conv3d(3, 3, (3, 7, 7), stride=1, padding=0)
input = autograd.Variable(torch.randn(1, 3, 7, 60, 40))
output = m(input)
print(output.size())
# 输出是 torch.Size([1, 3, 5, 54, 34])
