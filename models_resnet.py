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

####################################### ResNet ###########################################
class ResNet(nn.Module):

    def __init__(self, width, block, layers, num_class=1000, use_bn=True, eps_jl=0.5, keep_prob=None):
        widths = [64*width, 128*width, 256*width, 512*width]

        self.inplanes = 64
        super(ResNet, self).__init__()
        self.use_bn = use_bn
        self.keep_prob = keep_prob
        self.use_random_projection = False

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64, eps=0.001)
            self.bias = False
        else:
            self.bn1 = nn.Sequential()
            self.bias = True

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=self.bias)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group0 = self._make_layer(block, widths[0], layers[0], 'group0', use_bn=self.use_bn,
            keep_prob=keep_prob)
        self.group1 = self._make_layer(block, widths[1], layers[1], 'group1', use_bn=self.use_bn,
            keep_prob=keep_prob, stride=2, stride_conv2=2)
        self.group2 = self._make_layer(block, widths[2], layers[2], 'group2', use_bn=self.use_bn,
            keep_prob=keep_prob, stride=2)
        self.group3 = self._make_layer(block, widths[3], layers[3], 'group3', use_bn=self.use_bn,
            keep_prob=keep_prob, stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(widths[3] * block.expansion, num_class)

    def _make_layer(self, block, planes, blocks, group_name, stride=1, stride_conv2=1, use_bn=True, keep_prob=None):
        layers = []
        block_name = '{}-block{}'.format(group_name, 0)
        layers.append(block(self.inplanes, planes, block_name, stride=stride, stride_conv2=stride_conv2, use_bn=use_bn, keep_prob=keep_prob))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            block_name = '{}-block{}'.format(group_name, i)
            layers.append(block(self.inplanes, planes, block_name, use_bn=use_bn, keep_prob=keep_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def init_rp(self):
        pass              
    
    def setup_rp(self):
        pass

    def saving_fmaps(self):
        for m in self.modules():
            if isinstance(m, bb.BasicBlock):
                m.saving_fmaps()


class ResNetRP(nn.Module):

    def __init__(self, width, block, layers, num_class=1000, use_bn=True, eps_jl=0.5, keep_prob=None):
        widths = [64*width, 128*width, 256*width, 512*width]

        self.inplanes = 64
        super(ResNetRP, self).__init__()
        self.use_bn = use_bn
        self.keep_prob = keep_prob
        self.use_random_projection = False

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(64, eps=0.001)
            self.bias = False
        else:
            self.bn1 = nn.Sequential()
            self.bias = True

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=self.bias)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.group0 = self._make_layer(block, widths[0], layers[0], use_bn=self.use_bn,
            eps_jl=eps_jl, keep_prob=self.keep_prob)
        self.group1 = self._make_layer(block, widths[1], layers[1], use_bn=self.use_bn, stride=2, 
            eps_jl=eps_jl, keep_prob=self.keep_prob)
        self.group2 = self._make_layer(block, widths[2], layers[2], use_bn=self.use_bn, stride=2, 
            eps_jl=eps_jl, keep_prob=self.keep_prob)
        self.group3 = self._make_layer(block, widths[3], layers[3], use_bn=self.use_bn, stride=2,
            eps_jl=eps_jl, keep_prob=self.keep_prob)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(widths[3] * block.expansion, num_class)


    def _make_layer(self, block, planes, blocks, stride=1, use_bn=True, eps_jl=0.5, keep_prob=None):
        layers = []
        layers.append(block(self.inplanes, planes, None, stride=stride, use_bn=use_bn,
                            eps_jl=eps_jl, keep_prob=keep_prob))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, None, use_bn=use_bn, eps_jl=eps_jl, keep_prob=keep_prob))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.group0(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def init_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.generating_proj_matrix()                
    
    def setup_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.projecting_weight()



def resnet18(use_rp=False, width=1, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if use_rp:
        print('model using random projection')
        model = ResNetRP(width, bb.BasicBlockRP, [2, 2, 2, 2], **kwargs)
    else:
        # if args.keep_prob: print('model using top-k')
        model = ResNet(width, bb.BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

class ResNet152(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet152, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet152(use_rp=False, width=1, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if not use_rp:
        model = ResNet152(bb.Bottleneck, [3, 8, 36, 3], **kwargs)
    
    return model

######################################### ONLY for ResNet-8 ###############################################
class ResNet8(nn.Module):
    def __init__(self, width, block, num_class=10, use_bn=True, eps_jl=None, keep_prob=None):
        super(ResNet8, self).__init__()
        widths = [128*width, 256*width, 512*width]

        self.conv1 = bb.BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1)
        if use_bn:
            self.bn1 = nn.BatchNorm2d(64, eps=0.001)
            self.bias = False
        else:
            self.bn1 = nn.Sequential()
            self.bias = True
        # self.bn1 = nn.BatchNorm2d(64)

        self.block1 = block( 64, widths[0], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
        self.block2 = block(widths[0], widths[1], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
        self.block3 = block(widths[1], widths[2], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)

        self.classifier = nn.Sequential(
            nn.Linear(4*4*widths[2], 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_class))

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out, [0 for i in range(6)]

    def init_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.generating_proj_matrix()    

    def setup_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.projecting_weight()

    def turnoff_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.turnoff_rp()

  
def resnet8(use_rp=False, width=1, **kwargs):
    if use_rp:
        print('Using sparse random projection')
        model = ResNet8(width, bb.BasicBlockRP, **kwargs)
    else:
        model = ResNet8(width, bb.BasicBlock, **kwargs)
    return model


##################################### Wide ResNet ############################################
# WRN-28-10
class WRN(nn.Module):
    def __init__(self, width, block, num_class=10, use_bn=True, eps_jl=None, keep_prob=None):
        super(WRN, self).__init__()

        widths = [16 * width, 32 * width, 64 * width]

        self.conv1 = bb.BasicConv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.group0 = nn.Sequential(
                block(       16, widths[0], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[0], widths[0], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[0], widths[0], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[0], widths[0], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
            )

        self.group1 = nn.Sequential(
                block(widths[0], widths[1], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[1], widths[1], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[1], widths[1], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[1], widths[1], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
            )

        self.group2 = nn.Sequential(
                block(widths[1], widths[2], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[2], widths[2], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[2], widths[2], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[2], widths[2], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
            )

        self.classifier = nn.Sequential(
                nn.Linear(4*4*widths[2], 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, num_class))

    def forward(self, x):
        out = self.conv1(x)
        # print(out.size())
        out = F.relu(out)
        out = self.bn1(out)
        out = self.group0(out)
        # print(out.size())
        out = self.group1(out)
        # print(out.size())
        out = self.group2(out)
        # print(out.size())
        # out = F.max_pool2d(out, 8)

        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        # print(out.size())

        return out, [0 for i in range(6)]

    def init_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.generating_proj_matrix()    

    def setup_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.projecting_weight()



def wide_resnet(use_rp=False, **kwargs):
    if use_rp:
        print('Using sparse random projection')
        model = WRN(10, bb.BasicBlockRP, **kwargs)
    else:
        model = WRN(10, bb.BasicBlock, **kwargs)
    return model

class ResNet20(nn.Module):
    def __init__(self, width, block, num_class=10, use_bn=True, eps_jl=None, keep_prob=None):
        super(ResNet20, self).__init__()

        widths = [128 * width, 256 * width, 512 * width]

        self.conv1 = bb.BasicConv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.group0 = nn.Sequential(
                block(       64, widths[0], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                # block(widths[0], widths[0], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                # block(widths[0], widths[0], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[0], widths[0], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
            )

        self.group1 = nn.Sequential(
                block(widths[0], widths[1], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[1], widths[1], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[1], widths[1], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[1], widths[1], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
            )

        self.group2 = nn.Sequential(
                block(widths[1], widths[2], None, stride=2, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                # block(widths[2], widths[2], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[2], widths[2], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl),
                block(widths[2], widths[2], None, stride=1, use_bn=use_bn, keep_prob=keep_prob, eps_jl=eps_jl)
            )

        self.classifier = nn.Sequential(
                nn.Linear(4*4*widths[2], 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(1024, num_class))

    def forward(self, x):
        out = self.conv1(x)
        # print(out.size())
        out = F.relu(out)
        out = self.bn1(out)
        out = self.group0(out)
        # print(out.size())
        out = self.group1(out)
        # print(out.size())
        out = self.group2(out)
        # print(out.size())
        # out = F.max_pool2d(out, 8)

        out = out.view(out.size(0), -1)
        # print(out.size())
        out = self.classifier(out)
        # print(out.size())

        return out, [0 for i in range(6)]

    def init_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.generating_proj_matrix()    

    def setup_rp(self):
        for m in self.modules():
            if isinstance(m, bb.RPConv2d):
                m.projecting_weight()

def resnet20(use_rp=False, **kwargs):
    if use_rp:
        print('Using sparse random projection')
        model = ResNet20(1, bb.BasicBlockRP, **kwargs)
    else:
        model = ResNet20(1, bb.BasicBlock, **kwargs)
    return model
