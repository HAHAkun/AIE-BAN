import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
import cv2
import numpy as np
from torchvision.transforms import transforms

from PIL import Image


class L_WC(nn.Module):

    def __init__(self):
        super(L_WC, self).__init__()

    def forward(self, x):
        Drg = torch.pow(x[:, 0, :, :] - x[:, 1, :, :], 2)
        Drb = torch.pow(x[:, 0, :, :] - x[:, 2, :, :], 2)
        Dgb = torch.pow(x[:, 1, :, :] - x[:, 2, :, :], 2)


        k = Drg + Drb + Dgb

        return k


class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class L_MF(nn.Module):

    def __init__(self):
        super(L_MF, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_00= torch.FloatTensor( [[1,0,0],[0,0,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_01 = torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_02 = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [1, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_10 = torch.FloatTensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_11 = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_12 = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_20 = torch.FloatTensor([[0, 0, 1], [0, 0, 0], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_21 = torch.FloatTensor([[0, 0, 0], [0, 0, 1], [0, 0, 0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_22 = torch.FloatTensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]]).cuda().unsqueeze(0).unsqueeze(0)

        self.weight_00 = nn.Parameter(data=kernel_00, requires_grad=False)
        self.weight_01 = nn.Parameter(data=kernel_01, requires_grad=False)
        self.weight_02 = nn.Parameter(data=kernel_02, requires_grad=False)
        self.weight_10 = nn.Parameter(data=kernel_10, requires_grad=False)
        self.weight_11 = nn.Parameter(data=kernel_11, requires_grad=False)
        self.weight_12 = nn.Parameter(data=kernel_12, requires_grad=False)
        self.weight_20 = nn.Parameter(data=kernel_20, requires_grad=False)
        self.weight_21 = nn.Parameter(data=kernel_21, requires_grad=False)
        self.weight_22 = nn.Parameter(data=kernel_22, requires_grad=False)


    def forward(self, org  ):


        # org_mean = torch.mean(org,1,keepdim=True)
        org_R = org[:, 0:1, :, :]
        org_G = org[:, 1:2, :, :]
        org_B = org[:, 2:3, :, :]



        D_00_R = F.conv2d(org_R, self.weight_00, padding=1)
        D_01_R = F.conv2d(org_R, self.weight_01, padding=1)
        D_02_R = F.conv2d(org_R, self.weight_02, padding=1)
        D_10_R = F.conv2d(org_R, self.weight_10, padding=1)
        D_11_R = F.conv2d(org_R, self.weight_11, padding=1)
        D_12_R = F.conv2d(org_R, self.weight_12, padding=1)
        D_20_R = F.conv2d(org_R, self.weight_20, padding=1)
        D_21_R = F.conv2d(org_R, self.weight_21, padding=1)
        D_22_R = F.conv2d(org_R, self.weight_22, padding=1)
        D_all_R = torch.cat([D_00_R,D_01_R,D_02_R,D_10_R,D_11_R,D_12_R,D_20_R,D_21_R,D_22_R],dim=1)

        D_median_R = torch.median(D_all_R, 1, True)
        D_median_R = D_median_R[0]

        D_00_G = F.conv2d(org_G, self.weight_00, padding=1)
        D_01_G = F.conv2d(org_G, self.weight_01, padding=1)
        D_02_G = F.conv2d(org_G, self.weight_02, padding=1)
        D_10_G = F.conv2d(org_G, self.weight_10, padding=1)
        D_11_G = F.conv2d(org_G, self.weight_11, padding=1)
        D_12_G = F.conv2d(org_G, self.weight_12, padding=1)
        D_20_G = F.conv2d(org_G, self.weight_20, padding=1)
        D_21_G = F.conv2d(org_G, self.weight_21, padding=1)
        D_22_G = F.conv2d(org_G, self.weight_22, padding=1)
        D_all_G = torch.cat([D_00_G, D_01_G, D_02_G, D_10_G, D_11_G, D_12_G, D_20_G, D_21_G, D_22_G], dim=1)

        D_median_G = torch.median(D_all_G, 1, True)
        D_median_G = D_median_G[0]

        D_00_B = F.conv2d(org_B, self.weight_00, padding=1)
        D_01_B = F.conv2d(org_B, self.weight_01, padding=1)
        D_02_B = F.conv2d(org_B, self.weight_02, padding=1)
        D_10_B = F.conv2d(org_B, self.weight_10, padding=1)
        D_11_B = F.conv2d(org_B, self.weight_11, padding=1)
        D_12_B = F.conv2d(org_B, self.weight_12, padding=1)
        D_20_B = F.conv2d(org_B, self.weight_20, padding=1)
        D_21_B = F.conv2d(org_B, self.weight_21, padding=1)
        D_22_B = F.conv2d(org_B, self.weight_22, padding=1)
        D_all_B = torch.cat([D_00_B, D_01_B, D_02_B, D_10_B, D_11_B, D_12_B, D_20_B, D_21_B, D_22_B], dim=1)

        D_median_B = torch.median(D_all_B, 1, True)
        D_median_B = D_median_B[0]


        E = torch.pow(D_median_R-org_R, 2) + torch.pow(D_median_G-org_G, 2) + torch.pow(D_median_B-org_B, 2)


        return E


if __name__ == "__main__":
    device = torch.device('cuda:0')
    img = Image.open('data/test_data/DICM/01.jpg')
    img=img.resize((512,512))
    transf = transforms.ToTensor()

    img_tensor = transf(img).unsqueeze(0).to(device)
    ts = torch.ones([1,64*3,512,512]).to(device)
    # print(img_tensor.size())
    net = L_WC()

    net = net.to(device)

    # enhance_image_1, enhance_image, r, enhance_image_2 = net(img_tensor)
    enhanced = net(img_tensor)
    # print(img_tensor.size())
    # enhanced = enhanced[1]
    print(enhanced.size())