import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim_LLIE(img1, img2, img3, window, window_size, channel, e_mean=0.5, size_average=True,):
    # img1 is enhanced img, img2 is low-light img, img3 is HE-img, e_mean is
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu3 = F.conv2d(img3, window, padding=window_size // 2, groups=channel)
    e_mean = torch.ones(mu1.shape) * e_mean
    e_mean = e_mean.cuda(0)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu3_sq = mu3.pow(2)
    e_mean_sq = e_mean.pow(2)
    mu1_mu2 = mu1 * mu2
    mu1_mu3 = mu1 * mu3

    sigma1 = torch.abs(img1-mu1)
    sigma2 = torch.abs(img2-mu2)
    sigma3 = torch.abs(img3-mu3)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma3_sq = F.conv2d(img3 * img3, window, padding=window_size // 2, groups=channel) - mu3_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    l = (2 * mu1 * e_mean + C1) / (mu1_sq + e_mean_sq + C1)
    c = (2 * sigma1 * sigma3+C2) / (sigma1_sq + sigma3_sq + C2)
    s = (sigma12 + C2 / 2) / (sigma1*sigma2 + C2 / 2)

    ssim_map = l * c * s

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM_LLIE(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM_LLIE, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2, img3):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim_LLIE(img1, img2, img3, window, self.window_size, channel, self.size_average)


# def ssim_LLIE(img1, img2, img3, window_size=11, size_average=True):
#     (_, channel, _, _) = img1.size()
#     window = create_window(window_size, channel)
#
#     if img1.is_cuda:
#         window = window.cuda(img1.get_device())
#     window = window.type_as(img1)
#
#     return _ssim_LLIE(img1, img2, img3, window, window_size, channel, size_average)


if __name__ == '__main__':
    img1 = torch.rand(1, 1, 200, 200)
    img2 = torch.rand(1, 1, 200, 200)
    img3 = torch.rand(1, 1, 200, 200)

    img1 = img1.cuda(0)
    img2 = img2.cuda(0)
    img3 = img3.cuda(0)

    ssim_loss = SSIM_LLIE(window_size=11)

    print(ssim_loss(img1, img2, img3))