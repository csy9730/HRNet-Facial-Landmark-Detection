# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse
import cv2
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
#from lib.models.hrnet_jx import *
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function

import numpy as np
from PIL import Image
from torch.autograd import Variable
from lib.utils.transforms import fliplr_joints, crop, generate_target, transform_pixel
from lib.core.evaluation import decode_preds, compute_nme, get_preds
from lib.utils.transforms import transform_preds

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)
    #model = get_face_alignment_net(config)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()

    #device = torch.device("cpu")
    #model.to(device)


    # load model
    checkpoint = torch.load(args.model_file)
    #print(state_dict)

    model.load_state_dict(checkpoint)


    '''
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)
    '''

    


    #model.eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    #img = np.array(Image.open("outdoor_044.png").convert('RGB'), dtype=np.float32)
    img = np.array(Image.open("./data/300w/images/ibug/image_092.jpg").convert('RGB'), dtype=np.float32)
    
    scale = 0.32
    scale *= 1.25
    
    center = torch.Tensor([668.5, 205])
    

    img = crop(img, center, scale, [256,256], rot=0)
    raw_img = img


    print("img.shape:", img.shape)


    img = img.astype(np.float32)
    img = (img/255.0 - mean) / std
    img = img.transpose([2, 0, 1])

    input_tensor = torch.tensor(img)
    input_tensor = input_tensor.unsqueeze(0)

    output = model(input_tensor)
    score_map = output.data.cpu()
    print(score_map.shape)
    print(score_map.size(0))

    
    coords = get_preds(score_map)  # float type
    coords = coords.cpu()
    print("coords 1:", coords)
    print("coords.shape", coords.shape)


    for p in range(coords.size(1)):
        hm = output[0][p]
        px = int(math.floor(coords[0][p][0]))
        py = int(math.floor(coords[0][p][1]))
        if (px > 1) and (px < 64) and (py > 1) and (py < 64):
            diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
            coords[0][p] += diff.sign() * .25

    coords += 0.5
    print("coords 2:", coords)
    

    scale = 1.28
    center = torch.Tensor([128, 128])
    preds = transform_preds(coords[0], center, scale, [64,64])
    print("preds:", preds)
    print("preds shape:", preds.shape)


    for i in range(0,68):
        cv2.circle(raw_img,(int(preds[i][0]), int(preds[i][1])),2,(0,255,0),-1)


    cv2.imshow("1",raw_img)
    cv2.waitKey(0)
    

    #preds = decode_preds(score_map, center, [scale], [64, 64])
    #or n in range(score_map.size(0)):
    #    predictions[meta['index'][n], :, :] = preds[n, :, :]

    #nme, predictions = function.inference(config, test_loader, model)

    #torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))


def camera():
	args = parse_args()

	logger, final_output_dir, tb_log_dir = \
		utils.create_logger(config, args.cfg, 'test')

	logger.info(pprint.pformat(args))
	logger.info(pprint.pformat(config))

	cudnn.benchmark = config.CUDNN.BENCHMARK
	cudnn.determinstic = config.CUDNN.DETERMINISTIC
	cudnn.enabled = config.CUDNN.ENABLED

	config.defrost()
	config.MODEL.INIT_WEIGHTS = False
	config.freeze()
	model = models.get_face_alignment_net(config)
	#model = get_face_alignment_net(config)

	print("model :", model)

	gpus = list(config.GPUS)
	model = nn.DataParallel(model, device_ids=gpus)

	#device = torch.device("cpu")
	#model.to(device)


	# load model
	state_dict = torch.load(args.model_file)
	# keylist = list(state_dict.keys())
	# for i in keylist:
	#     ikey= i
	#     ikey= ikey.strip('module')
	#     ikey= ikey.strip('.')
	#     state_dict[ikey] = state_dict.pop(i)    
	#print(state_dict)

	model.load_state_dict(state_dict, strict=False)
	# for k,v in model.named_parameters():
	#     print("k:", k)
	#     if k == 'module.head.0.weight':
	#         print(v)
	# while 1:
	#     pass


	cap = cv2.VideoCapture(0)
	while True:
		ret, img = cap.read()
		if not ret:
			break
		#img = cv2.imread("1_0.jpg", 1)

		cut = img[0:480,80:560]
		height, width = cut.shape[:2]
		raw_img = cut

		input_image = cv2.resize(raw_img, (256,256))

		mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
		std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

		#print("input_image.shape:", input_image.shape)
		input_image = input_image.astype(np.float32)
		input_image = (input_image/255.0 - mean) / std
		input_image = input_image.transpose([2, 0, 1])

		input_tensor = torch.tensor(input_image)
		input_tensor = input_tensor.unsqueeze(0)
		#print("input_tensor.shape", input_tensor.shape)

		output = model(input_tensor)
		score_map = output.data.cpu()
		#print(score_map.shape)
		#print(score_map.size(0))


		coords = get_preds(score_map)  # float type
		coords = coords.cpu()
		#print("coords 1:", coords)
		#print("coords.shape", coords.shape)


		#print("height:", height)
		#print("width:", width)

		scale = 2.4
		center = torch.Tensor([240, 240])
		preds = transform_preds(coords[0], center, scale, [64,64])

		for i in range(0,68):
			cv2.circle(raw_img,(int(preds[i][0]), int(preds[i][1])),2,(0,255,0),-1)

		cv2.imshow("img", raw_img)
		cv2.imwrite("ditou.jpg", raw_img)
		cv2.waitKey(30)


if __name__ == '__main__':
    main()
    #camera()

