import operator
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import Parameter
from torch.autograd import Variable
from torch.autograd.function import InplaceFunction

import pyinn as P
import scipy.io as sio
import numpy as np

import sklearn.random_projection as rp
# import matplotlib as mp 
# import matplotlib.pyplot as plt

def my_mask(input, mask, p=0.5, training=False, inplace=False):
    # p is the probability of each hidden neuron being dropped
    return MyMaskLayer.apply(input, mask, p, training, inplace)

class MyMaskLayer(InplaceFunction):

    @staticmethod
    def _make_mask(m, p):
        return m.div_(p)

    @classmethod
    def forward(cls, ctx, input, mask, p, train=False, inplace=False):
        assert input.size() == mask.size()
        if p < 0 or p > 1:
            raise ValueError("Drop probability has to be between 0 and 1, "
                            "but got {}".format(p))

        ctx.p = p
        ctx.train = train
        ctx.inplace = inplace

        if ctx.p == 0 or not ctx.train:
            return input

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.mask = mask
        if ctx.p == 1:
            ctx.mask.fill_(0)
        # print('mask: ', ctx.mask)

        output.mul_(ctx.mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.mask)), None, None, None, None
        else:
            return grad_output, None, None, None, None
        

def topk_mask_gen(x, top):
    assert x.dim() == 4
    # extract mini-batch size m
    # print('x size: ', x.size())
    m = x.size(0)
    k = int(top * x.size(1) * x.size(2) * x.size(3))
    # k = int(top * x.size(0) * x.size(1) * x.size(2) * x.size(3))
    # print('k: ', k)
    mask = Variable(x.data, requires_grad=False)

    # for sample_idx in range(m):
    # approximately choose the top-k based on the first sample
    mask_topk = torch.topk(mask[0].view(-1), k, sorted=False)
    threshold = float(torch.min(mask_topk[0]))
    mask = F.threshold(mask, threshold, 0)
    mask = mask.abs()
    mask = torch.sign(mask)

    return mask


################################ Conv2D for Random Projection ###############################
class RPConv2d(nn.Module):
    """" Dynamic 2D convolution based on dense gemm and using batch matrix multiplication """
        # slice_size is CRS

    def __init__(self, in_channels, out_channels,
                kernel_size=3, stride=1, padding=0, use_bn=True,
                enable_RP=False, epsilon_JL=0.5, target_dim='auto', keep_prob=0.5):
        super(RPConv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # self.weight = Parameter(torch.Tensor(
            # out_channels, in_channels, kernel_size, kernel_size))
        self.use_bn = use_bn
        
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
            self.bias = None
        else:
            self.bn = nn.Sequential()
            self.bias = True
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=self.bias)
        
        self.pruning = False
        self.enable_RP = enable_RP
        if in_channels <= 3:
            self.enable_RP = False

        self.slice_size = int(kernel_size * kernel_size * in_channels)
        self.epsilon_JL = epsilon_JL
        self._k = target_dim
        self.keep_prob = keep_prob
        self.projector = None
        self.w_reduced = None

        self.fmaps = {'mask': [], 'fmap': []}
        self.save_mask = False

        # self.init_params()

    def forward(self, x):
        # y = F.conv2d(x, self.weight, bias=self.bias, stride=self.stride, 
        #         padding=self.padding)
        y = self.conv(x)

        y_o = 0
        active_prob = 0
        # if self.training and self.enable_RP:
        if self.enable_RP:
            x_col, size = self.im2col(x)
            # print(size)
            x_col = torch.transpose(x_col, 1, 2)
            # print('x_reduced: ', x_reduced.size())        
            # x_reduced = self.reduce_dim(x_reduced.data)
            # print('x_col: ', x_col.size())
            x_reduced = self.transform(x_col.data)
            # print('x_reduced: ', x_reduced.size())

            mask = self.generate_mask(x_reduced, topn=self.keep_prob)
            # if mask.dim() == 2:
            #     active_prob = float(torch.sum(mask) / (mask.size(0) * mask.size(1)))
            # elif mask.dim() == 3:
            #     active_prob = float(torch.sum(mask) / (mask.size(0) * mask.size(1) * mask.size(2)))
            # else:
            #     raise ValueError("masks should have a dim same as output activation")

            # print('active_prob: ', active_prob)
            # transform the original y to 2d matrix
            assert y.dim() == 4
            rs = y.size(2)
            y = y.view(y.size(0), y.size(1), rs * rs)
            assert y.size() == mask.size()
            y = my_mask(y, mask, p=1-self.keep_prob, training=True, inplace=False)
            y = F.relu(y, inplace=True)
            y = self.bn(y)
            y = my_mask(y, mask, p=1-self.keep_prob, training=True, inplace=False)
            if y.dim() != 4:
                y = y.view(y.size(0), y.size(1), rs, rs)

            if self.save_mask:
                self.fmaps['mask'].append(mask.data.cpu().numpy())
                self.fmaps['fmap'].append(y.data.cpu().numpy())
                print('mask being saved: ', mask.size())
                sio.savemat('results/{}'.format('vgg8_mask'), self.fmaps, do_compression=True)
                self.save_mask = False
        else:
            y = F.relu(y, inplace=True)
            y = self.bn(y)

        # print('after pruning: ', y[0])
        # self.plt_activations(y)

        # print(y.size())
        # return y, active_prob, y_o
        return y

    def saving_mask(self):
        self.save_mask = True
               
    def generate_mask(self, input, topn=None):
        mask = Variable(torch.matmul(input, self.w_reduced.transpose(0, 1)), requires_grad=False)
        assert mask.dim() == 3
        mask = mask.transpose(1, 2)
        mask = mask.contiguous()
        mask_slice = mask[0].view(-1)
        # print('mask slice: ', mask_slice)
        
        if topn == None:
            mask = F.relu(torch.sign(mask))
        else:
            mask_topn = torch.topk(mask_slice, int(topn * mask_slice.size(0)), dim=0, sorted=False)
            threshold = float(torch.min(mask_topn[0]))
            mask = F.threshold(mask, threshold, 0)
            mask = mask.abs()
            mask = torch.sign(mask)
        # print(mask)
        return mask

    def start_pruning(self):
        self.pruning = True

    def stop_pruning(self):
        self.pruning = False

    def turnoff_rp(self):
        self.enable_RP = False

    def projecting_weight(self):
        if self.enable_RP:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    self.w_reduced = self.transform(m.weight.data.view(m.weight.size()[0], -1))
            # print('w_reduced: ', self.w_reduced.shape)

    def generating_proj_matrix(self):
        if self.enable_RP:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    self.fit(m.weight.data.view(m.weight.size(0), -1))
                    

    def im2col(self, x):
        """ transform a 4D tensor of size [N,C,H,W] to [N, CRS, PQ]
            where filter size is [K,C,R,S] and output size is [N,K,P,Q]
            P and Q are calculated based on kernel size (R,S) and stride
        """
        assert x.dim() == 4
        x = P.im2col(x, self.kernel_size, self.stride, self.padding)
        size = x.size()
        new_size = [size[0], size[1]*size[2]*size[3], size[4]*size[5]]
        x = x.view(new_size)
        # print('x (col) size: ', x.size())
        return x, new_size

    def init_params(self):
        # if self.use_bn:
        #     import scipy.stats as stats
        #     stddev = self.stddev if hasattr(self, 'stddev') else 0.1
        #     X = stats.truncnorm(-2, 2, scale=stddev)
        #     values = torch.Tensor(X.rvs(self.weight.data.numel()))
        #     values = values.view(self.weight.data.size())
        #     self.weight.data.copy_(values)
        # else:
        #     nn.init.xavier_uniform(self.weight)
        
        # if self.bias is not None:
        #     self.bias.data.zero_()
        pass            

    def fit(self, X):
        # input X should be an matrix of n rows (n samples) and d columns (original dim d)
        assert X.dim() == 2

        n, d = X.shape

        if self._k == 'auto':
            self.k = self.calc_min_dim(n_samples=n, eps=self.epsilon_JL)
        else:
            self.k = self._k 

        if self.k <= 0:
            raise ValueError("Target dimension is invalid, got %d" % self.k)

        # elif self.k >= d:
            # print("Target dimensionis greater than original dimension")

        print("# of samples: {}, target dim: {}, original dim: {}, epsilon: {}".format(
                n, self.k, d, self.epsilon_JL))

        # density controlled by s
        s = 3
        np_sparse_matrix = np.random.binomial(1, float(1 / s), size=(d, self.k)) / math.sqrt(s)
        signs = np.random.binomial(1, 0.5, size=(d, self.k)) * 2 - 1
        np_sparse_matrix = np_sparse_matrix * signs
        _sparse_proj_matrix = torch.FloatTensor(np_sparse_matrix)
        # print(_sparse_proj_matrix)
        # print('positive ratio: ', np.count_nonzero(np.clip(np_sparse_matrix, 0, 10)) / (self.k * d))
        # print('negative ratio: ', np.count_nonzero(np.clip(np_sparse_matrix, -10, 0)) / (self.k * d))

        # _proj_matrix = torch.randn(d, self.k) / math.sqrt(self.k)
        self.register_buffer('proj_matrix', _sparse_proj_matrix)

    def transform(self, X):
        #  X could be 2D matrix or 3D tensor, but the last dimension should match d
        assert X.size(-1) == self.proj_matrix.size(0)
        projections = torch.matmul(X, self.proj_matrix)
        return projections

    def calc_min_dim(self, n_samples, eps):
        if eps <= 0.0 or eps >= 1:
            raise ValueError(
                "The eps for JL lemma should in [0, 1], got %r" % eps)
        if n_samples <=0:
            raise ValueError(
                "The number of samples should be greater than zero, got %r" % n_samples)
        denominator = (eps ** 2 / 2) - (eps ** 3 / 3)
        return int(4 * np.log(n_samples) / denominator)


class BasicConv2d(nn.Module):
    """ Basic conv2d with batch normalization and in-place ReLU activation 
        IMPORTANT bn is applied after ReLU
    """
    def __init__(self, in_channels, out_channels, use_bn=True, enable_RP=False, epsilon_JL=None, **kwargs):
        super(BasicConv2d, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
            self.bias = None
        else:
            self.bn = nn.Sequential()
            self.bias = True
                    
        self.conv = nn.Conv2d(in_channels, out_channels, bias=self.bias, **kwargs)
        # self.init_params()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        x = self.bn(x)
        return x

    def init_params(self):
        if not self.use_bn:
            nn.init.xavier_uniform(self.conv.weight)


class TopkConv2d(nn.Module):
    """ Conv2d with Top-K masking and batch normalization and in-place ReLU activation """
    def __init__(self, in_channels, out_channels, use_bn=True, keep_prob=None, enable_RP=False, epsilon_JL=None, **kwargs):
        super(TopkConv2d, self).__init__()
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
            self.bias = None
        else:
            self.bn = nn.Sequential()
            self.bias= True

        self.conv = nn.Conv2d(in_channels, out_channels, bias=self.bias, **kwargs)
        self.keep_prob = keep_prob
        # self.init_params()

    def forward(self, x):
        x = self.conv(x)
        if self.keep_prob:
            mask = topk_mask_gen(x, self.keep_prob)
            x = my_mask(x, mask, training=True)
            x = F.relu(x, inplace=True) 
            x = self.bn(x)
            x = my_mask(x, mask, training=True)
        else:
            x = F.relu(x, inplace=True)
            x = self.bn(x)
        return x

    def init_params(self):
        if not self.use_bn:
            nn.init.xavier_uniform(self.conv.weight)     


class SignConv2d(nn.Module):
    """ Conv2d with sign masking and batch normalization and in-place ReLU activation """
    def __init__(self, in_channels, out_channels,
            kernel_size=3, stride=1, padding=0, use_bn=True, 
            enable=True, keep_prob=None, **kwargs):
        super(SignConv2d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bn = use_bn
        
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
            self.bias = None
        else:
            self.bn = nn.Sequential()
            self.bias = True
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=self.bias)
        
        # self.pruning = False
        self.enable = enable
        self.slice_size = int(kernel_size * kernel_size * in_channels)
        # self.epsilon_JL = epsilon_JL
        # self._k = target_dim
        self.keep_prob = keep_prob
        self.projector = None
        self.w_reduced = None

    def forward(self, x):
        y = self.conv(x)

        active_prob = 0
        if self.enable:
        # if self.enable_RP:
            x_col, size = self.im2col(x)
            x_col = torch.transpose(x_col, 1, 2)
            # x_reduced = self.transform(x_col.data)
            x_reduced = torch.sign(x_col.data)

            mask = self.generate_mask(x_reduced, topn=self.keep_prob)
            assert y.dim() == 4
            rs = y.size(2)
            y = y.view(y.size(0), y.size(1), rs * rs)
            assert y.size() == mask.size()
            y = my_mask(y, mask, p=1-self.keep_prob, training=True, inplace=False)
            y = F.relu(y, inplace=True)
            y = self.bn(y)
            y = my_mask(y, mask, p=1-self.keep_prob, training=True, inplace=False)
            if y.dim() != 4:
                y = y.view(y.size(0), y.size(1), rs, rs)
        else:
            y = F.relu(y, inplace=True)
            y = self.bn(y)

        return y

    def projecting_weight(self):
        if self.enable:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # self.w_reduced = self.transform(m.weight.data.view(m.weight.size()[0], -1))
                    self.w_reduced = torch.sign(m.weight.data.view(m.weight.size()[0], -1))
               
    def generate_mask(self, input, topn=None):
        mask = Variable(torch.matmul(input, self.w_reduced.transpose(0, 1)), requires_grad=False)
        assert mask.dim() == 3
        mask = mask.transpose(1, 2)
        mask = mask.contiguous()
        mask_slice = mask[0].view(-1)
        
        if topn == None:
            mask = F.relu(torch.sign(mask))
        else:
            mask_topn = torch.topk(mask_slice, int(topn * mask_slice.size(0)), dim=0, sorted=False)
            threshold = float(torch.min(mask_topn[0]))
            mask = F.threshold(mask, threshold, 0)
            mask = mask.abs()
            mask = torch.sign(mask)

        # print('mask: ', torch.nonzero(mask.data).size(0) / (mask.size(0) * mask.size(1) * mask.size(2)))
        return mask

    def im2col(self, x):
        """ transform a 4D tensor of size [N,C,H,W] to [N, CRS, PQ]
            where filter size is [K,C,R,S] and output size is [N,K,P,Q]
            P and Q are calculated based on kernel size (R,S) and stride
        """
        assert x.dim() == 4
        x = P.im2col(x, self.kernel_size, self.stride, self.padding)
        size = x.size()
        new_size = [size[0], size[1]*size[2]*size[3], size[4]*size[5]]
        x = x.view(new_size)
        # print('x (col) size: ', x.size())
        return x, new_size        


################################# for ResNet blocks #########################################
class Conv2d(nn.Module):
    """ Basic conv2d """
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)

    def forward(self, x):
        return self.conv(x)


class BasicDense(nn.Module):
    """ Basic fully-connected layer with batch normalization and in-place ReLU"""
    def __init__(self, in_features, out_features):
        super(BasicDense, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.bn = nn.BatchNorm1d(out_features)
        # self.init_params()

    def forward(self, x):
        # print("BasicDense weight: ", self.weight.size())
        # print("BasicDense bias: ", self.bias.size())
        # print("BasicDense input: ", x.size())
        x = torch.matmul(x, torch.transpose(self.weight, 0, 1)) + self.bias
        x = self.bn(x)
        return x
    
    def init_params(self):
        nn.init.xavier_uniform(self.weight)
        self.bias.data.zero_()        


################################ Residual Block ################################################
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, name, stride=1, stride_conv2=1, use_bn=True, keep_prob=None, eps_jl=0.5):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bias = False
        else:
            self.bn1 = nn.Sequential()
            self.bn2 = nn.Sequential()
            self.bias = True
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=self.bias)

        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride_conv2, padding=1,
            bias=self.bias)

        self.stride = stride
        self.keep_prob = keep_prob

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, 
                    kernel_size=1, stride=stride*stride_conv2, padding=0))
        else:
            self.shortcut = nn.Sequential()

        self.name = name
        self.save_fmaps = False
        self.fmaps = {
            'conv1-before-bn': [],
            'conv2-before-bn': [],
            'conv1-after-bn': [],
            'conv2-after-bn': []
        }

    def forward(self, x):
        out = self.conv1(x)

        if self.keep_prob:
            mask = topk_mask_gen(out, self.keep_prob)
            out = my_mask(out, mask, training=True)
            out = F.relu(out)
            out = self.bn1(out)
            out = my_mask(out, mask, training=True)
        else:
            out = F.relu(out)
            # if self.save_fmaps:
            #     self.fmaps['conv1-before-bn'].append(out.data.cpu().numpy())
            out = self.bn1(out)
            # if self.save_fmaps:
            #     self.fmaps['conv1-after-bn'].append(out.data.cpu().numpy())
            

        out = self.conv2(out)

        if self.keep_prob:
            mask = topk_mask_gen(out, self.keep_prob)
            out = my_mask(out, mask, training=True)
            out = F.relu(out, inplace=True)
            out = self.bn2(out)
            out = my_mask(out, mask, training=True)
        else:
            out = F.relu(out)
            if self.save_fmaps:
                self.fmaps['conv2-before-bn'].append(out.data.cpu().numpy())
            out = self.bn2(out)
            if self.save_fmaps:
                self.fmaps['conv2-after-bn'].append(out.data.cpu().numpy())
                sio.savemat('results/{}'.format(self.name), self.fmaps, do_compression=True)
                self.save_fmaps = False
                
        out += self.shortcut(x)

        return out

    def saving_fmaps(self):
        self.save_fmaps = True


################################ Residual Block with Random Projection ###############################
class BasicBlockRP(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, name, stride=1, use_bn=True, 
        keep_prob=None, eps_jl=0.5):
        super(BasicBlockRP, self).__init__()
        self.use_bn = use_bn
        
        if self.use_bn:
            # self.bn1 = nn.BatchNorm2d(planes)
            # self.bn2 = nn.BatchNorm2d(planes)
            self.bias = False
        else:
            # self.bn1 = nn.Sequential()
            # self.bn2 = nn.Sequential()
            self.bias = True

        self.conv1 = RPConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                use_bn=use_bn, enable_RP=True, epsilon_JL=eps_jl, keep_prob=keep_prob)
        self.conv2 = RPConv2d(planes, planes, kernel_size=3, stride=1, padding=1,
                use_bn=use_bn, enable_RP=True, epsilon_JL=eps_jl, keep_prob=keep_prob)

        self.stride = stride
        self.keep_prob = keep_prob

        if stride != 1 or in_planes != self.expansion * planes:
            shortcut_conv = RPConv2d(in_planes, self.expansion * planes, 
                        kernel_size=1, stride=stride, padding=0,
                        use_bn=use_bn, enable_RP=False, keep_prob=keep_prob)
            self.shortcut = nn.Sequential(shortcut_conv)
        else:
            shortcut_conv = None
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)[0]      

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = RPConv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = RPConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = RPConv2d(planes, planes * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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
