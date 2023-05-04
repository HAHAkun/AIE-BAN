import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

device = torch.device('cuda:0')

class Retinex_block(nn.Module):

    def __init__(self):
        super(Retinex_block, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(number_f, 1, 3, 1, 1, bias=True)




    def forward(self, x):
        x = self.relu(self.conv1(x))
        # p1 = self.maxpool(x1)
        x = self.relu(self.conv2(x))
        # p2 = self.maxpool(x2)
        x = self.relu(self.conv3(x))

        return x


class Base_Net(nn.Module):

    def __init__(self):
        super(Base_Net, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        number_f = 32
        self.base_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.base_conv2 = nn.Conv2d(2*number_f, number_f, 3, 1, 1, bias=True)
        self.base_conv3 = nn.Conv2d(2*number_f, number_f, 3, 1, 1, bias=True)
        self.base_conv4 = nn.Conv2d(3+number_f, number_f, 3, 1, 1, bias=True)
        self.base_conv5 = nn.Conv2d(number_f, 3, 3, 1, 1, bias=True)
        self.cond_conv1 = nn.Conv2d(1, 3, 3, 1, 1, bias=True)
        self.cond_conv2 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.cond_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)

        self.Ini_block = Retinex_block()


    def forward(self, x):
        ill = torch.max(x, dim=1, keepdim=True)[0]
        y = self.relu(self.cond_conv1(ill))
        # x1 = y * x
        x1 = self.relu(self.base_conv1(x))
        y = self.relu(self.cond_conv2(y))
        x2 = y * x1
        x2 = torch.cat([x1, x2], dim=1)
        x2 = self.relu(self.base_conv2(x2))
        y = self.relu(self.cond_conv3(y))
        x3 = y * x2
        x3 = torch.cat([x, x3], dim=1)
        x3 = self.relu(self.base_conv4(x3))
        Weight_map = self.relu(self.base_conv5(x3))

        output = Weight_map * torch.pow(x, 0.4) + (1 - Weight_map) * torch.pow(x, 2.5)

        return ill, Weight_map, output





if __name__ == "__main__":
    device = torch.device('cuda:0')
    img = Image.open('data/test_data/DICM/01.jpg')
    img=img.resize((512,512))
    transf = transforms.ToTensor()

    img_tensor = transf(img).unsqueeze(0).to(device)
    ts = torch.ones([1,64*3,512,512]).to(device)
    # print(img_tensor.size())
    net = Base_Net()

    net = net.to(device)

    # enhance_image_1, enhance_image, r, enhance_image_2 = net(img_tensor)
    _,_,enhanced = net(img_tensor)
    # print(img_tensor.size())
    # enhanced = enhanced[1]
    print(enhanced.size())