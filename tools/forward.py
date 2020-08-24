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
    parser.add_argument('--model-in','-m', dest="model_file",help='model parameters', required=True, type=str)

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

    print("model :", model)

    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus)

    #device = torch.device("cpu")
    #model.to(device)


    # load model
    checkpoint = torch.load(args.model_file)
    state_dict = checkpoint['state_dict']

    # keylist = list(state_dict.keys())
    # for i in keylist:
    #     ikey= i
    #     ikey= ikey.strip('module')
    #     ikey= ikey.strip('.')
    #     state_dict[ikey] = state_dict.pop(i)    
    #print(state_dict)

    #model.load_state_dict(state_dict, strict=False)
    model.load_state_dict(state_dict)


    #cap = cv2.VideoCapture(0)
    #while True:
        #ret, img = cap.read()
        #if not ret:
        #    break
    
    img = cv2.imread("1.jpg", 1)

    #cut = img[0:480,80:560]
    #cut = img 
    height, width = img.shape[:2]
    raw_img = img

    input_image = cv2.resize(raw_img, (256,256))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)


    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    #std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

    #print("input_image.shape:", input_image.shape)
    input_image = input_image.astype(np.float32)
    input_image = (input_image/255.0 - mean) / std
    input_image = input_image.transpose([2, 0, 1])

    input_tensor = torch.tensor(input_image)
    input_tensor = input_tensor.unsqueeze(0)
    #print("input_tensor.shape", input_tensor.shape)

    output = model(input_tensor)
    score_map = output.data.cpu()
    #print("score_map:", score_map)
    #print("score_map.shape:", score_map.shape)
    #print(score_map.shape)
    #print(score_map.size(0))


    coords = get_preds(score_map)  # float type
    coords = coords.cpu()

    #print("coords.shape", coords.shape)


    #print("height:", height)
    #print("width:", width)

    scale = 2.4
    center = torch.Tensor([240, 240])
    preds = transform_preds(coords[0], center, scale, [64,64])

    for i in range(0,68):
        cv2.circle(raw_img,(int(preds[i][0]), int(preds[i][1])),2,(0,255,0),-1)
    #ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie, image_points)


    cv2.imshow("img", raw_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
    #camera()

