import os
import argparse
import shutil
import time
import json
import math
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.nn import Parameter
from torch.autograd import Variable

import scipy.io as sio
import numpy as np

import building_blocks as bb


# Model description
class VGG8(nn.Module):
    def __init__(self, num_class=10, use_bn=True, keep_prob=None):
        super(VGG8, self).__init__()

        k = 1
        self.conv1 = bb.BasicConv2d(      3, 128 * k, use_bn=use_bn, kernel_size=3, stride=1, padding=1)
        self.conv2 = bb.BasicConv2d(128 * k, 128 * k, use_bn=use_bn, kernel_size=3, stride=1, padding=1)
        self.conv3 = bb.BasicConv2d(128 * k, 256 * k, use_bn=use_bn, kernel_size=3, stride=1, padding=1)
        self.conv4 = bb.BasicConv2d(256 * k, 256 * k, use_bn=use_bn, kernel_size=3, stride=1, padding=1)
        self.conv5 = bb.BasicConv2d(256 * k, 512 * k, use_bn=use_bn, kernel_size=3, stride=1, padding=1) 
        self.conv6 = bb.BasicConv2d(512 * k, 512 * k, use_bn=use_bn, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(4*4*512*k, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_class)
        )

        if keep_prob:
            self.drop_prob = 1 - keep_prob
        else:
            self.drop_prob = 0

    def forward(self, x):
        x = self.conv1(x)
        # x = F.dropout(x, p=self.drop_prob, training=self.training)
        # a = F.threshold(x, 0.2, 0)
        # a = x
        # print('nonzero ratio: ', torch.nonzero(a.data).size(0) / (x.size(0) * x.size(1) * x.size(2) * x.size(3)))
        x = self.conv2(x)
        # x = F.dropout(x, p=self.drop_prob, training=self.training)
        # a = F.threshold(x, 0.2, 0)
        # a = x
        # print('nonzero ratio: ', torch.nonzero(a.data).size(0) / (x.size(0) * x.size(1) * x.size(2) * x.size(3)))
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        # x = F.dropout(x, p=self.drop_prob, training=self.training)
        # a = F.threshold(x, 0.2, 0)
        # a = x
        # print('nonzero ratio: ', torch.nonzero(a.data).size(0) / (x.size(0) * x.size(1) * x.size(2) * x.size(3)))
        x = self.conv4(x)
        # x = F.dropout(x, p=self.drop_prob, training=self.training)
        # a = F.threshold(x, 0.2, 0)
        # a = x
        # print('nonzero ratio: ', torch.nonzero(a.data).size(0) / (x.size(0) * x.size(1) * x.size(2) * x.size(3)))
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        # x = F.dropout(x, p=self.drop_prob, training=self.training)
        # a = F.threshold(x, 0.2, 0)
        # a = x
        # print('nonzero ratio: ', torch.nonzero(a.data).size(0) / (x.size(0) * x.size(1) * x.size(2) * x.size(3)))
        x = self.conv6(x)
        # x = F.dropout(x, p=self.drop_prob, training=self.training)
        # a = F.threshold(x, 0.2, 0)
        # a = x
        # print('nonzero ratio: ', torch.nonzero(a.data).size(0) / (x.size(0) * x.size(1) * x.size(2) * x.size(3)))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, [0 for i in range(6)]

class VGG8TopK(nn.Module):
    def __init__(self, num_class=10, use_bn=True, keep_prob=None):
        super(VGG8TopK, self).__init__()

        self.conv1 = bb.TopkConv2d(  3, 128, use_bn=use_bn, keep_prob=keep_prob, kernel_size=3, stride=1, padding=1)
        self.conv2 = bb.TopkConv2d(128, 128, use_bn=use_bn, keep_prob=keep_prob, kernel_size=3, stride=1, padding=1)
        self.conv3 = bb.TopkConv2d(128, 256, use_bn=use_bn, keep_prob=keep_prob, kernel_size=3, stride=1, padding=1)
        self.conv4 = bb.TopkConv2d(256, 256, use_bn=use_bn, keep_prob=keep_prob, kernel_size=3, stride=1, padding=1)
        self.conv5 = bb.TopkConv2d(256, 512, use_bn=use_bn, keep_prob=keep_prob, kernel_size=3, stride=1, padding=1) 
        self.conv6 = bb.TopkConv2d(512, 512, use_bn=use_bn, keep_prob=keep_prob, kernel_size=3, stride=1, padding=1)

        self.classifier = nn.Sequential(
            nn.Linear(4*4*512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)       
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # active_prob = [conv1_prob, conv2_prob, conv3_prob, conv4_prob, conv5_prob, conv6_prob]
        return x, [0 for i in range(6)]


class VGG8RP(nn.Module):
    def __init__(self, num_class=10, use_bn=True, keep_prob=None, eps_jl=0.5):
        super(VGG8RP, self).__init__()

        k = 1
        self.conv1 = bb.RPConv2d(    3, 128*k, kernel_size=3, stride=1, padding=1, 
            enable_RP=False)

        self.conv2 = bb.RPConv2d(128*k, 128*k, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable_RP=True, epsilon_JL=eps_jl, keep_prob=keep_prob)

        self.conv3 = bb.RPConv2d(128*k, 256*k, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable_RP=True, epsilon_JL=eps_jl, keep_prob=keep_prob)

        self.conv4 = bb.RPConv2d(256*k, 256*k, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable_RP=True, epsilon_JL=eps_jl, keep_prob=keep_prob)

        self.conv5 = bb.RPConv2d(256*k, 512*k, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable_RP=True, epsilon_JL=eps_jl, keep_prob=keep_prob) 

        self.conv6 = bb.RPConv2d(512*k, 512*k, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable_RP=True, epsilon_JL=eps_jl, keep_prob=keep_prob)


        self.classifier = nn.Sequential(
            nn.Linear(4*4*512*k, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        active_prob = []
        return x, active_prob

    def init_rp(self):
        self.conv1.generating_proj_matrix()
        self.conv2.generating_proj_matrix()
        self.conv3.generating_proj_matrix()
        self.conv4.generating_proj_matrix()
        self.conv5.generating_proj_matrix()
        self.conv6.generating_proj_matrix()

    def setup_rp(self):
        self.conv1.projecting_weight()
        self.conv2.projecting_weight()
        self.conv3.projecting_weight()
        self.conv4.projecting_weight()
        self.conv5.projecting_weight()
        self.conv6.projecting_weight()

    def turnoff_rp(self):
        self.conv1.turnoff_rp()
        self.conv2.turnoff_rp()
        self.conv3.turnoff_rp()
        self.conv4.turnoff_rp()
        self.conv5.turnoff_rp()
        self.conv6.turnoff_rp()

    def saving_mask(self):
        self.conv4.saving_mask()

    def start_collecting(self):
        self.collecting = True

class VGG8Sign(nn.Module):
    def __init__(self, num_class=10, use_bn=True, keep_prob=None, eps_jl=0.5):
        super(VGG8Sign, self).__init__()

        self.conv1 = bb.SignConv2d(  3, 128, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable=True, keep_prob=keep_prob)

        self.conv2 = bb.SignConv2d(128, 128, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable=True, keep_prob=keep_prob)

        self.conv3 = bb.SignConv2d(128, 256, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable=True, keep_prob=keep_prob)

        self.conv4 = bb.SignConv2d(256, 256, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable=True, keep_prob=keep_prob)

        self.conv5 = bb.SignConv2d(256, 512, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable=True, keep_prob=keep_prob) 

        self.conv6 = bb.SignConv2d(512, 512, kernel_size=3, stride=1, padding=1, 
            use_bn=use_bn, enable=True, keep_prob=keep_prob)


        self.classifier = nn.Sequential(
            nn.Linear(4*4*512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_class)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = self.conv6(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # active_prob = [conv1_prob, conv2_prob, conv3_prob, conv4_prob, conv5_prob, conv6_prob]
        return x, [0 for i in range(6)]

    def setup_rp(self):
        self.conv1.projecting_weight()
        self.conv2.projecting_weight()
        self.conv3.projecting_weight()
        self.conv4.projecting_weight()
        self.conv5.projecting_weight()
        self.conv6.projecting_weight()


class AlexNet(nn.Module):

    def __init__(self, num_class=1000, use_bn=True):
        super(AlexNet, self).__init__()
    
        self.conv1 = bb.RPConv2d(3, 64, kernel_size=11, stride=4, padding=2)
        self.conv2 = bb.RPConv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = bb.RPConv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = bb.RPConv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = bb.RPConv2d(256, 256, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu =  nn.ReLU(inplace=True)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(64, eps=0.001)
            self.bn2 = nn.BatchNorm2d(192, eps=0.001)
            self.bn3 = nn.BatchNorm2d(384, eps=0.001)
            self.bn4 = nn.BatchNorm2d(256, eps=0.001)
            self.bn5 = nn.BatchNorm2d(256, eps=0.001)
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()
            self.bn3 = nn.Sequential()
            self.bn4 = nn.Sequential()
            self.bn5 = nn.Sequential()

        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_class),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.bn3(x)

        x = self.conv4(x)
        x = self.relu(x)
        x = self.bn4(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn5(x)
        x = self.maxpool(x)

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x