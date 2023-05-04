import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dateloader_train
import data
import Myloss
import numpy as np
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

from models import Base_Net
from NR_SSIM import SSIM_LLIE
device = torch.device('cuda:0')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)





def train(config):

	os.environ['CUDA_VISIBLE_DEVICES']='0'

	BAN = Base_Net().cuda()

	BAN.apply(weights_init)
	if config.load_pretrain == True:
	    BAN.load_state_dict(torch.load(config.pretrain_dir))
	# train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
	train_dataset = dateloader_train.lowlight_loader(config.lowlight_images_path)

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	Loss_MF = Myloss.L_MF()
	Loss_WC = Myloss.L_WC()
	Loss_TV = Myloss.L_TV()
	L_ssim = SSIM_LLIE(11)



	optimizer = torch.optim.Adam(BAN.parameters(), lr=config.lr, weight_decay=config.weight_decay)



	BAN.train()

	for epoch in range(config.num_epochs):
		print('__________________第{}轮开始_______________'.format(epoch))
		for iteration, imgs in enumerate(train_loader):
			img_lowlight, img_HE = imgs
			img_lowlight = img_lowlight.cuda()
			img_HE = img_HE.cuda()
			illumination, weight_map, enhanced_image= BAN(img_lowlight)

			loss_RTV = 80 * torch.mean(Loss_MF(enhanced_image))
			loss_TV = 80 * Loss_TV(weight_map)
			loss_col2 = 10 * torch.mean(Loss_WC(weight_map))
			loss_ssim = 10 * (1-L_ssim(enhanced_image, img_lowlight, img_HE))

			loss = loss_col2 + loss_TV + loss_ssim + loss_RTV
			
			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(BAN.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("Loss at iteration", iteration+1, ":", loss.item())

			if ((iteration+1) % config.snapshot_iter) == 0:

				torch.save(BAN.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')





if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--train_batch_size', type=int, default=32)
	parser.add_argument('--val_batch_size', type=int, default=16)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=10)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--load_pretrain', type=bool, default=False)
	parser.add_argument('--pretrain_dir', type=str, default= "snapshots/Epoch15.pth")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)


	train(config)








	
