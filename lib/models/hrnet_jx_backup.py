# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Create by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion,
                               momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(num_channels[branch_index] * block.expansion,
                            momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    # nn.Upsample(scale_factor=2**(j-i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                BatchNorm2d(num_outchannels_conv3x3,
                                            momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[x[i].shape[2], x[i].shape[3]],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, config, **kwargs):
        self.inplanes = 64
        extra = config.MODEL.EXTRA
        super(HighResolutionNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.sf = nn.Softmax(dim=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # stage 2
        self.stage2_basic_1_1 = BasicBlock(18,18,1)
        self.stage2_basic_1_2 = BasicBlock(18,18,1)
        self.stage2_basic_1_3 = BasicBlock(18,18,1)
        self.stage2_basic_1_4 = BasicBlock(18,18,1)


        self.stage2_basic_2_1 = BasicBlock(36,36,1)
        self.stage2_basic_2_2 = BasicBlock(36,36,1)
        self.stage2_basic_2_3 = BasicBlock(36,36,1)
        self.stage2_basic_2_4 = BasicBlock(36,36,1)

        self.stage2_36_18 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage2_18_36 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))


        # stage 3
        self.stage3_basic_1_1 = BasicBlock(18,18,1)
        self.stage3_basic_1_2 = BasicBlock(18,18,1)
        self.stage3_basic_1_3 = BasicBlock(18,18,1)
        self.stage3_basic_1_4 = BasicBlock(18,18,1)
        self.stage3_basic_1_5 = BasicBlock(18,18,1)
        self.stage3_basic_1_6 = BasicBlock(18,18,1)
        self.stage3_basic_1_7 = BasicBlock(18,18,1)
        self.stage3_basic_1_8 = BasicBlock(18,18,1)
        self.stage3_basic_1_9 = BasicBlock(18,18,1)
        self.stage3_basic_1_10 = BasicBlock(18,18,1)
        self.stage3_basic_1_11 = BasicBlock(18,18,1)
        self.stage3_basic_1_12 = BasicBlock(18,18,1)
        self.stage3_basic_1_13 = BasicBlock(18,18,1)
        self.stage3_basic_1_14 = BasicBlock(18,18,1)
        self.stage3_basic_1_15 = BasicBlock(18,18,1)
        self.stage3_basic_1_16 = BasicBlock(18,18,1)

        self.stage3_basic_2_1 = BasicBlock(36,36,1)
        self.stage3_basic_2_2 = BasicBlock(36,36,1)
        self.stage3_basic_2_3 = BasicBlock(36,36,1)
        self.stage3_basic_2_4 = BasicBlock(36,36,1)
        self.stage3_basic_2_5 = BasicBlock(36,36,1)
        self.stage3_basic_2_6 = BasicBlock(36,36,1)
        self.stage3_basic_2_7 = BasicBlock(36,36,1)
        self.stage3_basic_2_8 = BasicBlock(36,36,1)
        self.stage3_basic_2_9 = BasicBlock(36,36,1)
        self.stage3_basic_2_10 = BasicBlock(36,36,1)
        self.stage3_basic_2_11 = BasicBlock(36,36,1)
        self.stage3_basic_2_12 = BasicBlock(36,36,1)
        self.stage3_basic_2_13 = BasicBlock(36,36,1)
        self.stage3_basic_2_14 = BasicBlock(36,36,1)
        self.stage3_basic_2_15 = BasicBlock(36,36,1)
        self.stage3_basic_2_16 = BasicBlock(36,36,1)

        self.stage3_basic_3_1 = BasicBlock(72,72,1)
        self.stage3_basic_3_2 = BasicBlock(72,72,1)
        self.stage3_basic_3_3 = BasicBlock(72,72,1)
        self.stage3_basic_3_4 = BasicBlock(72,72,1)
        self.stage3_basic_3_5 = BasicBlock(72,72,1)
        self.stage3_basic_3_6 = BasicBlock(72,72,1)
        self.stage3_basic_3_7 = BasicBlock(72,72,1)
        self.stage3_basic_3_8 = BasicBlock(72,72,1)
        self.stage3_basic_3_9 = BasicBlock(72,72,1)
        self.stage3_basic_3_10 = BasicBlock(72,72,1)
        self.stage3_basic_3_11 = BasicBlock(72,72,1)
        self.stage3_basic_3_12 = BasicBlock(72,72,1)
        self.stage3_basic_3_13 = BasicBlock(72,72,1)
        self.stage3_basic_3_14 = BasicBlock(72,72,1)
        self.stage3_basic_3_15 = BasicBlock(72,72,1)
        self.stage3_basic_3_16 = BasicBlock(72,72,1)

        self.stage3_36_18_1 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_36_18_2 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_36_18_3 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_36_18_4 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_72_18_1 = nn.Sequential(nn.Conv2d(72,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_72_18_2 = nn.Sequential(nn.Conv2d(72,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_72_18_3 = nn.Sequential(nn.Conv2d(72,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_72_18_4 = nn.Sequential(nn.Conv2d(72,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage3_72_36_1 = nn.Sequential(nn.Conv2d(72,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage3_72_36_2 = nn.Sequential(nn.Conv2d(72,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage3_72_36_3 = nn.Sequential(nn.Conv2d(72,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage3_72_36_4 = nn.Sequential(nn.Conv2d(72,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage3_18_18_1 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage3_18_18_2 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage3_18_18_3 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage3_18_18_4 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))

        self.stage3_18_36_1 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage3_18_36_2 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage3_18_36_3 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage3_18_36_4 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))

        self.stage3_18_72_1 = nn.Sequential(nn.Conv2d(18,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage3_18_72_2 = nn.Sequential(nn.Conv2d(18,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage3_18_72_3 = nn.Sequential(nn.Conv2d(18,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage3_18_72_4 = nn.Sequential(nn.Conv2d(18,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        
        self.stage3_36_72_1 = nn.Sequential(nn.Conv2d(36,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage3_36_72_2 = nn.Sequential(nn.Conv2d(36,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage3_36_72_3 = nn.Sequential(nn.Conv2d(36,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage3_36_72_4 = nn.Sequential(nn.Conv2d(36,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))

        # stage 4
        self.stage4_basic_1_1 = BasicBlock(18,18,1)
        self.stage4_basic_1_2 = BasicBlock(18,18,1)
        self.stage4_basic_1_3 = BasicBlock(18,18,1)
        self.stage4_basic_1_4 = BasicBlock(18,18,1)
        self.stage4_basic_1_5 = BasicBlock(18,18,1)
        self.stage4_basic_1_6 = BasicBlock(18,18,1)
        self.stage4_basic_1_7 = BasicBlock(18,18,1)
        self.stage4_basic_1_8 = BasicBlock(18,18,1)
        self.stage4_basic_1_9 = BasicBlock(18,18,1)
        self.stage4_basic_1_10 = BasicBlock(18,18,1)
        self.stage4_basic_1_11 = BasicBlock(18,18,1)
        self.stage4_basic_1_12 = BasicBlock(18,18,1)

        self.stage4_basic_2_1 = BasicBlock(36,36,1)
        self.stage4_basic_2_2 = BasicBlock(36,36,1)
        self.stage4_basic_2_3 = BasicBlock(36,36,1)
        self.stage4_basic_2_4 = BasicBlock(36,36,1)
        self.stage4_basic_2_5 = BasicBlock(36,36,1)
        self.stage4_basic_2_6 = BasicBlock(36,36,1)
        self.stage4_basic_2_7 = BasicBlock(36,36,1)
        self.stage4_basic_2_8 = BasicBlock(36,36,1)
        self.stage4_basic_2_9 = BasicBlock(36,36,1)
        self.stage4_basic_2_10 = BasicBlock(36,36,1)
        self.stage4_basic_2_11 = BasicBlock(36,36,1)
        self.stage4_basic_2_12 = BasicBlock(36,36,1)

        self.stage4_basic_3_1 = BasicBlock(72,72,1)
        self.stage4_basic_3_2 = BasicBlock(72,72,1)
        self.stage4_basic_3_3 = BasicBlock(72,72,1)
        self.stage4_basic_3_4 = BasicBlock(72,72,1)
        self.stage4_basic_3_5 = BasicBlock(72,72,1)
        self.stage4_basic_3_6 = BasicBlock(72,72,1)
        self.stage4_basic_3_7 = BasicBlock(72,72,1)
        self.stage4_basic_3_8 = BasicBlock(72,72,1)
        self.stage4_basic_3_9 = BasicBlock(72,72,1)
        self.stage4_basic_3_10 = BasicBlock(72,72,1)
        self.stage4_basic_3_11 = BasicBlock(72,72,1)
        self.stage4_basic_3_12 = BasicBlock(72,72,1)

        self.stage4_basic_4_1 = BasicBlock(144,144,1)
        self.stage4_basic_4_2 = BasicBlock(144,144,1)
        self.stage4_basic_4_3 = BasicBlock(144,144,1)
        self.stage4_basic_4_4 = BasicBlock(144,144,1)
        self.stage4_basic_4_5 = BasicBlock(144,144,1)
        self.stage4_basic_4_6 = BasicBlock(144,144,1)
        self.stage4_basic_4_7 = BasicBlock(144,144,1)
        self.stage4_basic_4_8 = BasicBlock(144,144,1)
        self.stage4_basic_4_9 = BasicBlock(144,144,1)
        self.stage4_basic_4_10 = BasicBlock(144,144,1)
        self.stage4_basic_4_11 = BasicBlock(144,144,1)
        self.stage4_basic_4_12 = BasicBlock(144,144,1)

        self.stage4_18_18_1 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_2 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_3 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_4 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_5 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_6 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_7 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_8 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_18_18_9 = nn.Sequential(nn.Conv2d(18,18, 3, 2,1,bias=False),
                        BatchNorm2d(18, momentum=0.01), nn.ReLU(inplace=True))

        self.stage4_18_36_1 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage4_18_36_2 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage4_18_36_3 = nn.Sequential(nn.Conv2d(18,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01))

        self.stage4_18_72_1 = nn.Sequential(nn.Conv2d(18,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage4_18_72_2 = nn.Sequential(nn.Conv2d(18,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage4_18_72_3 = nn.Sequential(nn.Conv2d(18,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))

        self.stage4_18_144_1 = nn.Sequential(nn.Conv2d(18,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))
        self.stage4_18_144_2 = nn.Sequential(nn.Conv2d(18,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))
        self.stage4_18_144_3 = nn.Sequential(nn.Conv2d(18,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))

        self.stage4_36_18_1 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage4_36_18_2 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage4_36_18_3 = nn.Sequential(nn.Conv2d(36,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))

        self.stage4_36_36_1 = nn.Sequential(nn.Conv2d(36,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_36_36_2 = nn.Sequential(nn.Conv2d(36,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01), nn.ReLU(inplace=True))
        self.stage4_36_36_3 = nn.Sequential(nn.Conv2d(36,36, 3, 2,1,bias=False),
                        BatchNorm2d(36, momentum=0.01), nn.ReLU(inplace=True))

        self.stage4_36_72_1 = nn.Sequential(nn.Conv2d(36,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage4_36_72_2 = nn.Sequential(nn.Conv2d(36,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage4_36_72_3 = nn.Sequential(nn.Conv2d(36,72, 3, 2,1,bias=False),
                        BatchNorm2d(72, momentum=0.01))

        self.stage4_36_144_1 = nn.Sequential(nn.Conv2d(36,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))
        self.stage4_36_144_2 = nn.Sequential(nn.Conv2d(36,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))
        self.stage4_36_144_3 = nn.Sequential(nn.Conv2d(36,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))

        self.stage4_72_18_1 = nn.Sequential(nn.Conv2d(72,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage4_72_18_2 = nn.Sequential(nn.Conv2d(72,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage4_72_18_3 = nn.Sequential(nn.Conv2d(72,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))


        self.stage4_72_36_1 = nn.Sequential(nn.Conv2d(72,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage4_72_36_2 = nn.Sequential(nn.Conv2d(72,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage4_72_36_3 = nn.Sequential(nn.Conv2d(72,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))

        self.stage4_72_144_1 = nn.Sequential(nn.Conv2d(72,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))
        self.stage4_72_144_2 = nn.Sequential(nn.Conv2d(72,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))
        self.stage4_72_144_3 = nn.Sequential(nn.Conv2d(72,144, 3, 2,1,bias=False),
                        BatchNorm2d(144, momentum=0.01))

        self.stage4_144_18_1 = nn.Sequential(nn.Conv2d(144,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage4_144_18_2 = nn.Sequential(nn.Conv2d(144,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))
        self.stage4_144_18_3 = nn.Sequential(nn.Conv2d(144,18, 1, 1,0,bias=False),
                        BatchNorm2d(18, momentum=0.01))

        self.stage4_144_36_1 = nn.Sequential(nn.Conv2d(144,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage4_144_36_2 = nn.Sequential(nn.Conv2d(144,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))
        self.stage4_144_36_3 = nn.Sequential(nn.Conv2d(144,36, 1, 1,0,bias=False),
                        BatchNorm2d(36, momentum=0.01))

        self.stage4_144_72_1 = nn.Sequential(nn.Conv2d(144,72, 1, 1,0,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage4_144_72_2 = nn.Sequential(nn.Conv2d(144,72, 1, 1,0,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        self.stage4_144_72_3 = nn.Sequential(nn.Conv2d(144,72, 1, 1,0,bias=False),
                        BatchNorm2d(72, momentum=0.01))
        
        
        


        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        _, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        _, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        _, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        final_inp_channels = sum(pre_stage_channels)

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=final_inp_channels,
                kernel_size=1,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0),
            BatchNorm2d(final_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=final_inp_channels,
                out_channels=config.MODEL.NUM_JOINTS,
                kernel_size=extra.FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # h, w = x.size(2), x.size(3)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        ## stage 2
        x1= self.transition1[0](x)
        x2 = self.transition1[1](x)

        # stage 2 basicBlock
        x1 = self.stage2_basic_1_1(x1)
        x1 = self.stage2_basic_1_2(x1)
        x1 = self.stage2_basic_1_3(x1)
        x1 = self.stage2_basic_1_4(x1)

        x2 = self.stage2_basic_2_1(x2)
        x2 = self.stage2_basic_2_2(x2)
        x2 = self.stage2_basic_2_3(x2)
        x2 = self.stage2_basic_2_4(x2)

        # stage 2 fuse
        fuse1 = x1 + F.interpolate(self.stage2_36_18(x2),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = x2 + self.stage2_18_36(x1)
        fuse2 = self.relu(fuse2)

        ## stage 3 
        x1 = fuse1
        x2 = fuse2
        x3 = self.transition2[2](x2)

        # stage 3 modules 1
        # stage 3 basicBlock
        x1 = self.stage3_basic_1_1(x1)
        x1 = self.stage3_basic_1_2(x1)
        x1 = self.stage3_basic_1_3(x1)
        x1 = self.stage3_basic_1_4(x1)

        x2 = self.stage3_basic_2_1(x2)
        x2 = self.stage3_basic_2_2(x2)
        x2 = self.stage3_basic_2_3(x2)
        x2 = self.stage3_basic_2_4(x2)

        x3 = self.stage3_basic_3_1(x3)
        x3 = self.stage3_basic_3_2(x3)
        x3 = self.stage3_basic_3_3(x3)
        x3 = self.stage3_basic_3_4(x3)

        # stage 3 fuse
        fuse1 = x1 + F.interpolate(self.stage3_36_18_1(x2),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage3_72_18_1(x3),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = self.stage3_18_36_1(x1)
        fuse2 = fuse2 + F.interpolate(self.stage3_72_36_1(x3),size=[32,32], mode='bilinear')
        fuse2 = self.relu(fuse2)

        fuse3 = self.stage3_18_18_1(x1)
        fuse3 = self.stage3_18_72_1(fuse3)
        fuse3 = fuse3 + self.stage3_36_72_1(x2)
        fuse3 = fuse3 + x3
        fuse3 = self.relu(fuse3)

        x1 = fuse1
        x2 = fuse2
        x3 = fuse3

        # stage 3 modules 2
        # stage 3 basicBlock
        x1 = self.stage3_basic_1_5(x1)
        x1 = self.stage3_basic_1_6(x1)
        x1 = self.stage3_basic_1_7(x1)
        x1 = self.stage3_basic_1_8(x1)

        x2 = self.stage3_basic_2_5(x2)
        x2 = self.stage3_basic_2_6(x2)
        x2 = self.stage3_basic_2_7(x2)
        x2 = self.stage3_basic_2_8(x2)

        x3 = self.stage3_basic_3_5(x3)
        x3 = self.stage3_basic_3_6(x3)
        x3 = self.stage3_basic_3_7(x3)
        x3 = self.stage3_basic_3_8(x3)

        # stage 3 fuse
        fuse1 = x1 + F.interpolate(self.stage3_36_18_2(x2),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage3_72_18_2(x3),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = self.stage3_18_36_2(x1)
        fuse2 = fuse2 + F.interpolate(self.stage3_72_36_2(x3),size=[32,32], mode='bilinear')
        fuse2 = self.relu(fuse2)

        fuse3 = self.stage3_18_18_2(x1)
        fuse3 = self.stage3_18_72_2(fuse3)
        fuse3 = fuse3 + self.stage3_36_72_2(x2)
        fuse3 = fuse3 + x3
        fuse3 = self.relu(fuse3)

        x1 = fuse1
        x2 = fuse2
        x3 = fuse3


        # stage 3 modules 3
        # stage 3 basicBlock
        x1 = self.stage3_basic_1_9(x1)
        x1 = self.stage3_basic_1_10(x1)
        x1 = self.stage3_basic_1_11(x1)
        x1 = self.stage3_basic_1_12(x1)

        x2 = self.stage3_basic_2_9(x2)
        x2 = self.stage3_basic_2_10(x2)
        x2 = self.stage3_basic_2_11(x2)
        x2 = self.stage3_basic_2_12(x2)

        x3 = self.stage3_basic_3_9(x3)
        x3 = self.stage3_basic_3_10(x3)
        x3 = self.stage3_basic_3_11(x3)
        x3 = self.stage3_basic_3_12(x3)

        # stage 3 fuse
        fuse1 = x1 + F.interpolate(self.stage3_36_18_3(x2),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage3_72_18_3(x3),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = self.stage3_18_36_3(x1)
        fuse2 = fuse2 + F.interpolate(self.stage3_72_36_3(x3),size=[32,32], mode='bilinear')
        fuse2 = self.relu(fuse2)

        fuse3 = self.stage3_18_18_3(x1)
        fuse3 = self.stage3_18_72_3(fuse3)
        fuse3 = fuse3 + self.stage3_36_72_3(x2)
        fuse3 = fuse3 + x3
        fuse3 = self.relu(fuse3)

        x1 = fuse1
        x2 = fuse2
        x3 = fuse3

        # stage 3 modules 4
        # stage 3 basicBlock
        x1 = self.stage3_basic_1_13(x1)
        x1 = self.stage3_basic_1_14(x1)
        x1 = self.stage3_basic_1_15(x1)
        x1 = self.stage3_basic_1_16(x1)

        x2 = self.stage3_basic_2_13(x2)
        x2 = self.stage3_basic_2_14(x2)
        x2 = self.stage3_basic_2_15(x2)
        x2 = self.stage3_basic_2_16(x2)

        x3 = self.stage3_basic_3_13(x3)
        x3 = self.stage3_basic_3_14(x3)
        x3 = self.stage3_basic_3_15(x3)
        x3 = self.stage3_basic_3_16(x3)

        # stage 3 fuse
        fuse1 = x1 + F.interpolate(self.stage3_36_18_3(x2),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage3_72_18_4(x3),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = self.stage3_18_36_4(x1)
        fuse2 = fuse2 + F.interpolate(self.stage3_72_36_4(x3),size=[32,32], mode='bilinear')
        fuse2 = self.relu(fuse2)

        fuse3 = self.stage3_18_18_4(x1)
        fuse3 = self.stage3_18_72_4(fuse3)
        fuse3 = fuse3 + self.stage3_36_72_4(x2)
        fuse3 = fuse3 + x3
        fuse3 = self.relu(fuse3)

        x1 = fuse1
        x2 = fuse2
        x3 = fuse3

        # stage 4

        x4 = self.transition3[3](x3)

        # stage 4 modules 1
        # stage 4 basicBlock
        x1 = self.stage4_basic_1_1(x1)
        x1 = self.stage4_basic_1_2(x1)
        x1 = self.stage4_basic_1_3(x1)
        x1 = self.stage4_basic_1_4(x1)

        x2 = self.stage4_basic_2_1(x2)
        x2 = self.stage4_basic_2_2(x2)
        x2 = self.stage4_basic_2_3(x2)
        x2 = self.stage4_basic_2_4(x2)

        x3 = self.stage4_basic_3_1(x3)
        x3 = self.stage4_basic_3_2(x3)
        x3 = self.stage4_basic_3_3(x3)
        x3 = self.stage4_basic_3_4(x3)

        x4 = self.stage4_basic_4_1(x4)
        x4 = self.stage4_basic_4_2(x4)
        x4 = self.stage4_basic_4_3(x4)
        x4 = self.stage4_basic_4_4(x4)

        # stage 4 fuse
        fuse1 = x1 + F.interpolate(self.stage4_36_18_1(x2),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage4_72_18_1(x3),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage4_144_18_1(x4),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = self.stage4_18_36_1(x1)
        fuse2 = fuse2 + F.interpolate(self.stage4_72_36_1(x3),size=[32,32], mode='bilinear')
        fuse2 = fuse2 + F.interpolate(self.stage4_144_36_1(x4),size=[32,32], mode='bilinear')
        fuse2 = self.relu(fuse2)

        fuse3 = self.stage4_18_18_1(x1)
        fuse3 = self.stage4_18_72_1(fuse3)
        fuse3 = fuse3 +self.stage4_36_72_1(x2)
        fuse3 = fuse3 + x3
        fuse3 = fuse3 + F.interpolate(self.stage4_144_72_1(x4),size=[16,16], mode='bilinear')
        fuse3 = self.relu(fuse3)

        fuse4 = self.stage4_18_18_2(x1)
        fuse4 = self.stage4_18_18_3(fuse4)
        fuse4 = self.stage4_18_144_1(fuse4)
        fuse4 = fuse4 + self.stage4_36_144_1(self.stage4_36_36_1(x2))
        fuse4 = fuse4 + self.stage4_72_144_1(x3)
        fuse4 = fuse4 + x4

        x1 = fuse1
        x2 = fuse2
        x3 = fuse3
        x4 = fuse4

        # stage 4 modules 2
        # stage 4 basicBlock
        x1 = self.stage4_basic_1_5(x1)
        x1 = self.stage4_basic_1_6(x1)
        x1 = self.stage4_basic_1_7(x1)
        x1 = self.stage4_basic_1_8(x1)

        x2 = self.stage4_basic_2_5(x2)
        x2 = self.stage4_basic_2_6(x2)
        x2 = self.stage4_basic_2_7(x2)
        x2 = self.stage4_basic_2_8(x2)

        x3 = self.stage4_basic_3_5(x3)
        x3 = self.stage4_basic_3_6(x3)
        x3 = self.stage4_basic_3_7(x3)
        x3 = self.stage4_basic_3_8(x3)

        x4 = self.stage4_basic_4_5(x4)
        x4 = self.stage4_basic_4_6(x4)
        x4 = self.stage4_basic_4_7(x4)
        x4 = self.stage4_basic_4_8(x4)

        # stage 4 fuse
        fuse1 = x1 + F.interpolate(self.stage4_36_18_2(x2),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage4_72_18_2(x3),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage4_144_18_2(x4),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = self.stage4_18_36_2(x1)
        fuse2 = fuse2 + F.interpolate(self.stage4_72_36_2(x3),size=[32,32], mode='bilinear')
        fuse2 = fuse2 + F.interpolate(self.stage4_144_36_2(x4),size=[32,32], mode='bilinear')
        fuse2 = self.relu(fuse2)

        fuse3 = self.stage4_18_18_4(x1)
        fuse3 = self.stage4_18_72_2(fuse3)
        fuse3 = fuse3 +self.stage4_36_72_2(x2)
        fuse3 = fuse3 + x3
        fuse3 = fuse3 + F.interpolate(self.stage4_144_72_2(x4),size=[16,16], mode='bilinear')
        fuse3 = self.relu(fuse3)

        fuse4 = self.stage4_18_18_5(x1)
        fuse4 = self.stage4_18_18_6(fuse4)
        fuse4 = self.stage4_18_144_2(fuse4)
        fuse4 = fuse4 + self.stage4_36_144_2(self.stage4_36_36_2(x2))
        fuse4 = fuse4 + self.stage4_72_144_2(x3)
        fuse4 = fuse4 + x4

        x1 = fuse1
        x2 = fuse2
        x3 = fuse3
        x4 = fuse4

        # stage 4 modules 3
        # stage 4 basicBlock
        x1 = self.stage4_basic_1_9(x1)
        x1 = self.stage4_basic_1_10(x1)
        x1 = self.stage4_basic_1_11(x1)
        x1 = self.stage4_basic_1_12(x1)

        x2 = self.stage4_basic_2_9(x2)
        x2 = self.stage4_basic_2_10(x2)
        x2 = self.stage4_basic_2_11(x2)
        x2 = self.stage4_basic_2_12(x2)

        x3 = self.stage4_basic_3_9(x3)
        x3 = self.stage4_basic_3_10(x3)
        x3 = self.stage4_basic_3_11(x3)
        x3 = self.stage4_basic_3_12(x3)

        x4 = self.stage4_basic_4_9(x4)
        x4 = self.stage4_basic_4_10(x4)
        x4 = self.stage4_basic_4_11(x4)
        x4 = self.stage4_basic_4_12(x4)

        # stage 4 fuse
        fuse1 = x1 + F.interpolate(self.stage4_36_18_3(x2),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage4_72_18_3(x3),size=[64,64], mode='bilinear')
        fuse1 = fuse1 + F.interpolate(self.stage4_144_18_3(x4),size=[64,64], mode='bilinear')
        fuse1 = self.relu(fuse1)

        fuse2 = self.stage4_18_36_3(x1)
        fuse2 = fuse2 + F.interpolate(self.stage4_72_36_3(x3),size=[32,32], mode='bilinear')
        fuse2 = fuse2 + F.interpolate(self.stage4_144_36_3(x4),size=[32,32], mode='bilinear')
        fuse2 = self.relu(fuse2)

        fuse3 = self.stage4_18_18_7(x1)
        fuse3 = self.stage4_18_72_3(fuse3)
        fuse3 = fuse3 +self.stage4_36_72_3(x2)
        fuse3 = fuse3 + x3
        fuse3 = fuse3 + F.interpolate(self.stage4_144_72_3(x4),size=[16,16], mode='bilinear')
        fuse3 = self.relu(fuse3)

        fuse4 = self.stage4_18_18_8(x1)
        fuse4 = self.stage4_18_18_9(fuse4)
        fuse4 = self.stage4_18_144_3(fuse4)
        fuse4 = fuse4 + self.stage4_36_144_3(self.stage4_36_36_3(x2))
        fuse4 = fuse4 + self.stage4_72_144_3(x3)
        fuse4 = fuse4 + x4

        out0 = fuse1
        out1 = fuse2
        out2 = fuse3
        out3 = fuse4

        # # Head Part
        height, width = 64, 64
        x1 = F.interpolate(out1, size=(height, width), mode='bilinear', align_corners=False)
        x2 = F.interpolate(out2, size=(height, width), mode='bilinear', align_corners=False)
        x3 = F.interpolate(out3, size=(height, width), mode='bilinear', align_corners=False)
        x = torch.cat([out0, x1, x2, x3], 1)
        x = self.head(x)

        return x

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                logger.info(
                    '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_face_alignment_net(config, **kwargs):

    model = HighResolutionNet(config, **kwargs)
    pretrained = config.MODEL.PRETRAINED if config.MODEL.INIT_WEIGHTS else ''
    model.init_weights(pretrained=pretrained)

    return model

