import torch
import torch.onnx
from torch.autograd import Variable

print(torch.__version__)
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
#import lib.models as models
from lib.models.hrnet_jx import *
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

    #model = models.get_face_alignment_net(config)
    model = get_face_alignment_net(config)

    gpus = list(config.GPUS)
    #model = nn.DataParallel(model, device_ids=gpus).cuda()
    #model.load_state_dict(torch.load(args.model_file))
    #state_dict = torch.load(args.model_file, map_location='cpu')
    
    checkpoint = torch.load(args.model_file, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    keylist = list(state_dict.keys())
    for i in keylist:
        ikey= i
        ikey= ikey.strip('module')
        ikey= ikey.strip('.')
        state_dict[ikey] = state_dict.pop(i)   
    model.load_state_dict(state_dict, strict=False) 



    #model.load_state_dict(checkpoint)

    '''
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    input_image = np.random.rand(256,256,3)
    input_image = input_image.astype(np.float32)
    input_image = (input_image/255.0 - mean) / std
    input_image = input_image.transpose([2, 0, 1])

    input_tensor = torch.tensor(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    print("input_tensor:", input_tensor.shape)
    '''

    d_input = Variable(torch.randn(1,3,256,256))
    torch.onnx.export(model, d_input, './onnx_model/face_alignment.onnx', verbose=True, training=False)


main()

'''
from backbone import mobilenet
model = mobilenet.MobileFacenet()
model.load_state_dict(torch.load('./logs/mobilefaceNet_arcface/best.mobilenet.2019-09-05-4438.pth.tar', map_location='cpu'))

d_input = Variable(torch.randn(1,3,112,112))
torch.onnx.export(model, d_input, './onnx_model/MobileFacenet_temp.onnx', verbose=True)
'''