'''ResNet in PyTorch.'''
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
#from models import *
import scipy.io as sio
import numpy as np

# import building_blocks as bb
from models import *
from models_resnet import *
from init import LSUVinit

import gpustat

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')

parser.add_argument('--data-folder', '-d',
                    help='path to dataset')
parser.add_argument('--arch', default='None', type=str,
                    help='Specify model architecture')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=0.1, type=float,
                    metavar='LRD', help='lr decay (default: 0.1)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--plot-filename', default='plot', type=str,
                    help='Specify the filename of plot')
parser.add_argument('--ckpt-filename', default='checkpoint.pth.tar', type=str,
                    help='Specify the filename of checkpoint')
parser.add_argument('--fp16', dest='fp16', action='store_true',
                    help='use half-precision FP16')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='print shape of tensors')
parser.add_argument('--keep-prob', default=None, type=float, metavar='KEEP',
                    help='the ratio of kept (active) neurons')
parser.add_argument('--topk', dest='topk', action='store_true',
                    help='turn on top-k mask on CONV layer')
parser.add_argument('--sign-mask', dest='sign_mask', action='store_true',
                    help='turn on sign mask on CONV layer')
parser.add_argument('--random-proj', dest='rp', action='store_true',
                    help='turn on Random Projection based masking on CONV layer')
parser.add_argument('--turnoff-rp', dest='rp_off', action='store_true',
                    help='turn OFF Random Projection based masking on CONV layer')
parser.add_argument('--proj-update-freq', default=100, type=int,
                    help='Random Projection update frequency (default 100)')
parser.add_argument('--eps-jl', default=0.5, type=float,
                    metavar='EJL', help='epsilon_JL (default: 0.5)')
parser.add_argument('--no-bn', dest='bn', action='store_false',
                    help='turn off Batch Normalization')
parser.add_argument('--width', default=1, type=int,
                    help='Wide ResNet width (default 1)')
parser.add_argument('--num-class', default=10, type=int,
                    help='By default use CIFAR10')

def mem_usage(device=1):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    item = gpu_stats.jsonify()['gpus'][device]
    print('{}/{}, utilization {}'.format(item['memory.used'], item['memory.total'], item['utilization.gpu']))

def main():
    # Parse arguments
    global args, best_acc, use_cuda, dtype
    args = parser.parse_args()
    best_acc = 0
    data_dir = args.data_folder
    use_cuda = torch.cuda.is_available()
    dtype = torch.FloatTensor
    torch.manual_seed(args.seed)

    # Instantiate model
    if args.rp or args.topk:
        if not args.keep_prob: 
            print('Please specify keep_prob when using random projection or top-k')
            return
    print('Loading model')
    if args.arch == 'vgg8':
        if args.topk:
            model = VGG8TopK(num_class=args.num_class, use_bn=args.bn, keep_prob=args.keep_prob)
            print('model using VGG8TopK')
        elif args.rp:
            model = VGG8RP(num_class=args.num_class, use_bn=args.bn, keep_prob=args.keep_prob, 
                eps_jl=args.eps_jl)
            print('model using VGG8 with sparse random projection')
        elif args.sign_mask:
            model = VGG8Sign(num_class=args.num_class, use_bn=args.bn, keep_prob=args.keep_prob)
            print('model using sign mask')
        else:
            model = VGG8(num_class=args.num_class, use_bn=args.bn, keep_prob=args.keep_prob)
            print('model using basic VGG8')
    elif args.arch == 'resnet8':
        model = resnet8(num_class=args.num_class, use_bn=args.bn, use_rp=args.rp, eps_jl=args.eps_jl, 
            keep_prob=args.keep_prob, width=args.width)
        print('model using basic ResNet8')
        if args.keep_prob and not args.rp:
            print('Passing keep_prob w/o random-proj assumes using topk')
    elif args.arch == 'resnet20':
        model = resnet20(use_bn=args.bn, use_rp=args.rp, eps_jl=args.eps_jl, 
            keep_prob=args.keep_prob)
        print('model using ResNet-20')
        if args.keep_prob and not args.rp:
            print('Passing keep_prob w/o random-proj assumes using topk')            
    else:
        print('Please specify model by passing -a')
        return

    if not args.bn:
        print('model not using Batch Normalization')
    print('Finish load model')

    # Setup model and criterion if using GPU
    if use_cuda:
        dtype = torch.cuda.FloatTensor
        torch.cuda.manual_seed(args.seed)
        model.cuda()
        cudnn.benchmark = True
        criterion = nn.CrossEntropyLoss().cuda()
        print('Using CUDA')
    else:
        criterion = nn.CrossEntropyLoss()
        print('No CUDA involved')
    # if args.rp:
        # model.setup_rp()

    # Setup optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, 
        weight_decay=args.weight_decay, momentum=args.momentum)

    # Data preprocessing
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    if args.num_class == 100:
        trainset = dset.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                        shuffle=True, num_workers=args.workers)

        testset = dset.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=args.workers)
    else:
        trainset = dset.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, 
                        shuffle=True, num_workers=args.workers)

        testset = dset.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, 
                        shuffle=False, num_workers=args.workers)


    # Training/inference steps
    lr = args.lr
    lr_decay = args.lr_decay
    best_loss = 100000.0
    best_loss_epoch = 0
    lr_update_epoch = 0

    loss_plt = {'train': [], 'eval': []}
    acc_plt = {'train': [], 'eval': []}
    act_prob_plt = {
        'conv1': [],
        'conv2': [],
        'conv3': [],
        'conv4': [],
        'conv5': [],
        'conv6': []
    }

    # LSUV initialization
    if args.rp:
        model.init_rp()
        if use_cuda:
            model.cuda()
        model.setup_rp()

    if args.rp and args.rp_off:
        model.turnoff_rp()

    if args.sign_mask:
        model.setup_rp()

    if not (args.evaluate or args.resume):
        end = time.time()
        for i, (inputs, targets) in enumerate(train_loader):
            # measure LSUV initialization time
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs_var, targets_var = Variable(inputs), Variable(targets)
            model = LSUVinit(model, inputs_var, cuda=use_cuda)
            if i > 0: break
        print('LSUVinit time: ', time.time() - end)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    if args.evaluate:
        evaluate(test_loader, model, criterion)
    else:
        for epoch in range(args.start_epoch, args.epochs):
            # if args.distributed:
            #     train_sampler.set_epoch(epoch)
            lr = adjust_learning_rate(optimizer, lr, epoch + 1)
            # train for one epoch
            train_loss, train_acc, train_layer_act_prob = train(train_loader, model, 
                criterion, optimizer, lr, epoch)

            # evaluate on validation set
            eval_loss, eval_acc = evaluate(test_loader, model, criterion)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_loss_epoch = epoch

            # if epoch > best_loss_epoch + 10:
            #     if epoch > lr_update_epoch + 10:
            #         lr = lr * lr_decay
            #         lr_update_epoch = epoch

            # remember best eval_acc and save checkpoint
            is_best = eval_acc > best_acc
            best_acc = max(eval_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                # 'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, epoch, filename=args.ckpt_filename)

            loss_plt['train'].append(train_loss)
            loss_plt['eval'].append(eval_loss)
            acc_plt['train'].append(train_acc)
            acc_plt['eval'].append(eval_acc)
            # for l in range(1, 7):
            #     act_prob_plt['conv%d' % l].append(format(train_layer_act_prob['conv%d' % l].avg, '.2f'))

        f = open(args.plot_filename, 'w')
        json.dump({'train loss': loss_plt['train'],
                    'eval loss': loss_plt['eval'],
                    'train accuracy': acc_plt['train'],
                    'eval accuracy': acc_plt['eval'],
                    'Training active_prob: ': act_prob_plt
                    }, f)
        f.close()


# Training
def train(train_loader, model, criterion, optimizer, lr, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    layer_act_prob = {
        'conv1': AverageMeter(),
        'conv2': AverageMeter(),
        'conv3': AverageMeter(),
        'conv4': AverageMeter(),
        'conv5': AverageMeter(),
        'conv6': AverageMeter()
        }

    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        if args.fp16:
            inputs = inputs.cuda().half()
        
        if (args.rp or args.sign_mask) and i % args.proj_update_freq == 0:
            model.setup_rp()
            # print('Re-projecting weights')


        inputs_var, targets_var = Variable(inputs), Variable(targets)

        # compute output
        outputs, active_prob = model(inputs_var)
        loss = criterion(outputs, targets_var)

        # measure accuracy and record loss
        m_acc = accuracy(outputs.data, targets)
        losses.update(loss.data[0], inputs.size(0))
        acc.update(m_acc, inputs.size(0))
        # for l in range(1, 7):
        #     layer_act_prob['conv%d' % l].update(active_prob[l-1], inputs.size(0))
        
        # compute gradients and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure computation time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] '
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f}) '
              'Loss {loss.val:.4f} ({loss.avg:.4f}) '
              'Acc {acc.val:.3f} ({acc.avg:.3f}) '
              'lr {lr} '.format(
               epoch, i, len(train_loader), batch_time=batch_time,
               loss=losses, acc=acc, lr=lr))

    return losses.avg, acc.avg, layer_act_prob


def accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    batch_size = targets.size(0)
    correct = predicted.eq(targets).float().sum()
    return correct * (100.0 / batch_size)


def calc_error_rate(outputs, targets):
    return 1 - accuracy(outputs, targets)


def evaluate(data_loader, model, criterion):
    # global best_acc
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    if args.rp:
        model.setup_rp()

    end = time.time()
    for i, (inputs, targets) in enumerate(data_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        if args.fp16:
            inputs = inputs.cuda().half()
        inputs_var = torch.autograd.Variable(inputs, volatile=True)
        targets_var = torch.autograd.Variable(targets, volatile=True)

        if i == 10: model.saving_mask()

        # compute output and loss
        outputs, _ = model(inputs_var)
        m_loss = criterion(outputs, targets_var)

        # measure acc and record loss
        m_acc = accuracy(outputs.data, targets)
        # m_error_rate = calc_error_rate(outputs.data, targets)
        losses.update(m_loss.data[0], inputs.size(0))
        acc.update(m_acc, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy: {acc.val:.3f} ({acc.avg:.3f})\t'.format(
                   i, len(data_loader), batch_time=batch_time, loss=losses,
                   acc=acc))
    # model.save_fmaps(args.start_epoch)    
    return losses.avg, acc.avg


def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    bestname = 'best_' + filename 
    if is_best:
        shutil.copyfile(filename, bestname)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch in [100, 150, 200]:
        lr *= args.lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    main()