import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader_test

import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

from models import Base_Net
# from modelDfen import Base_Net

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(image_path)
	# data_lowlight = data_lowlight.resize((512, 512), Image.ANTIALIAS)

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	# factor = 8
	# h, w = data_lowlight.shape[2], data_lowlight.shape[3]
	# H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
	# padh = H - h if h % factor != 0 else 0
	# padw = W - w if w % factor != 0 else 0
	# data_lowlight = F.pad(data_lowlight, (0, padw, 0, padh), 'reflect')

	BAN = Base_Net().cuda()
	BAN.load_state_dict(torch.load('snapshots/Epoch99.pth'))
	start = time.time()
	i_image, m_image, enhanced_image = BAN(data_lowlight)


	end_time = (time.time() - start)
	print(end_time)
	image_path_g = image_path.replace('test_data', 'result_data')


	result_path = image_path_g

	if not os.path.exists(image_path_g.replace('/'+image_path_g.split("/")[-1],'')):
		os.makedirs(image_path_g.replace('/'+image_path_g.split("/")[-1],''))
	# Windows
	# if not os.path.exists(image_path_g.replace('\\'+image_path_g.split("\\")[-1],'')):
	# 	os.makedirs(image_path_g.replace('\\'+image_path_g.split("\\")[-1],''))
	torchvision.utils.save_image(enhanced_image, result_path)


if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'data/test_data/'
	
		file_list = os.listdir(filePath)

		for file_name in file_list:
			test_list = glob.glob(filePath+file_name+"/*") 
			for image in test_list:
				# image = image
				print(image)
				lowlight(image)

		

