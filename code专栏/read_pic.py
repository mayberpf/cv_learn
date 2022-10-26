# 使用PIL.Image读取图片
from re import S
import torch
from PIL import Image
import pdb
import torchvision.transforms as transforms
def image_proprecess(img_path):
    img = Image.open(img_path)
    data_transforms = transforms.Compose([
        # transforms.Resize((384, 384), interpolation=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    data = data_transforms(img)
    data = torch.unsqueeze(data,0)
    return img, data

image_path = '/home/ktd/rpf_ws/cv_learn/img/auto.png'
# pdb.set_trace()
img,data = image_proprecess(image_path)
print(img)
print(data)