import argparse
import os
import random
import shutil
import time
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import plainnet as pn
import resnet as rn
import densenet as dn

# used for logging to TensorBoard
from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=0.5, type=float,
                    help='compression rate in transition stage (default: 0.5)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')


def main():
    # random.seed(3423432)
    # torch.cuda.manual_seed(0)
    global args, best_acc, suffix

    datasize = [50000]
    args = parser.parse_args()
    # if args.tensorboard: configure("runs/%s" % (args.name))
    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
    transform_train = transforms.Compose([transforms.ToTensor(), normalize])
    dataset_train = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)
    noisy_data = add_noise(dataset_train, 0.1)
    print noisy_data

    # noise_levels = list(np.arange(0, 0.2, 0.02))

#     for size in datasize:
#         for noise in noise_levels:
#
#             noisy_data = add_noise(dataset_train, noise)
#
#             suffix = " - " + str(size)
#             tmp_train = random.sample(list(noisy_data), size)
#
#             tmp_test = dataset_test
#             kwargs = {'num_workers': 12, 'pin_memory': True}
#             train_loader = torch.utils.data.DataLoader(
#                 tmp_train,
#                 batch_size=args.batch_size, shuffle=True, **kwargs)
#             val_loader = torch.utils.data.DataLoader(
#                 tmp_test,
#                 batch_size=args.batch_size, shuffle=True, **kwargs)
#
#             n = 0
#             if args.mode == 2:
#                 depth = {20: 3, 32: 5, 44: 7, 56: 9, 68: 11, 80: 13, 92: 15, 104: 17, 110: 18, 1202: 200}
#                 n = depth[args.layers]
#
#             # create model
#             mode = {1: dn.DenseNet3(args.layers, 10, args.growth),
#                     2: resnet.ResNet(args.growth, [n, n, n]),
#                     3: plainnet.PlainNet(args.layers, 10, args.growth)}
#             print('Selected',)
#             print(mode[args.mode])
#             model = mode[args.mode]
#             # get the number of model parameters
#             print('Number of model parameters: {}'.format(
#                 sum([p.data.nelement() for p in model.parameters()])))
#             # for training on multiple GPUs.
#             # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
#             model = model.cuda()
#
#             cudnn.benchmark = True
#
#             # define loss function (criterion) and pptimizer
#             criterion = nn.CrossEntropyLoss().cuda()
#             optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                                         momentum=args.momentum,
#                                         weight_decay=args.weight_decay)
#
#             best_acc = 0
#             for epoch in tqdm.trange(args.start_epoch, args.epochs,
#                                      desc='Train'):
#                 adjust_learning_rate(optimizer, epoch)
#                 # train for one epoch
#                 acc_train = train(train_loader, model, criterion, optimizer, epoch)
#                 # evaluate on validation set
#                 acc_val = validate(val_loader, model, criterion, epoch)
#
#                 if args.tensorboard:
#                     log_value("generalization error" + suffix, acc_train - acc_val, epoch)
#
#                 # remember best precision and save checkpoint
#                 is_best = acc_val > best_acc
#                 best_acc = max(acc_val, best_acc)
#
#             print('Best accuracy' + suffix + ': ', best_acc)
#             if args.tensorboard:
#                 log_value('dataset accuracy', best_acc, size)
#
#
# def train(train_loader, model, criterion, optimizer, epoch):
#     """Train for one epoch on the training set"""
#     # batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     # switch to train mode
#     model.train()
#
#     # end = time.time()
#     for i, (input, target) in tqdm.tqdm(
#             enumerate(train_loader), total=len(train_loader),
#             desc='Train Iteration=%d' % epoch, leave=False):
#         target = target.cuda(async=True)
#         input = input.cuda()
#         input_var = torch.autograd.Variable(input)
#         target_var = torch.autograd.Variable(target)
#
#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)
#
#         # measure accuracy and record loss
#         prec1 = accuracy(output.data, target, topk=(1,))[0]
#         losses.update(loss.data[0], input.size(0))
#         # print(top1.avg, top1.count)
#         # print(prec1[0], input.size(0))
#         top1.update(prec1[0], input.size(0))
#
#         # compute gradient and do SGD step
#         optimizer.zero_grad()
#         loss.backward()
#
#         # add_regularization(model, args.batchnorm_decay, 1.0)
#         optimizer.step()
#
#         # measure elapsed time
#         # batch_time.update(time.time() - end)
#         # end = time.time()
#
#         # if i % args.print_freq == 0:
#         #     print('Epoch: [{0}][{1}/{2}]\t'
#         #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#         #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#         #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#         #               epoch, i, len(train_loader), batch_time=batch_time,
#         #               loss=losses, top1=top1))
#     # log to TensorBoard
#     if args.tensorboard:
#         log_value('train_loss' + suffix, losses.avg, epoch)
#         log_value('train_acc' + suffix, top1.avg, epoch)
#
#     return top1.avg
#
#
# def validate(val_loader, model, criterion, epoch):
#     """Perform validation on the validation set"""
#     # batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#
#     # end = time.time()
#     for i, (input, target) in tqdm.tqdm(
#             enumerate(val_loader), total=len(val_loader),
#             desc='Valid Iteration=%d' % epoch, leave=False):
#         target = target.cuda(async=True)
#         input = input.cuda()
#         input_var = torch.autograd.Variable(input, volatile=True)
#         target_var = torch.autograd.Variable(target, volatile=True)
#
#         # compute output
#         output = model(input_var)
#         loss = criterion(output, target_var)
#
#         # measure accuracy and record loss
#         prec1 = accuracy(output.data, target, topk=(1,))[0]
#         losses.update(loss.data[0], input.size(0))
#         # print(top1.avg, top1.count)
#         # print(prec1[0], input.size(0))
#
#         top1.update(prec1[0], input.size(0))
#
#         # measure elapsed time
#         # batch_time.update(time.time() - end)
#         # end = time.time()
#
#         # if i % args.print_freq == 0:
#         #     print('Test: [{0}/{1}]\t'
#         #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#         #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#         #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#         #               i, len(val_loader), batch_time=batch_time, loss=losses,
#         #               top1=top1))
#
#     # print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
#
#     # log to TensorBoard
#     if args.tensorboard:
#         log_value('val_loss' + suffix, losses.avg, epoch)
#         log_value('val_acc' + suffix, top1.avg, epoch)
#     return top1.avg
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
#     # log to TensorBoard
#     if args.tensorboard:
#         log_value('learning_rate' + suffix, lr, epoch)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# def accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
#
#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#     res = []
#     for k in topk:
#         correct_k = correct[:k].view(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res
#
#
def add_noise(dataset, noisy_rate):
    a = list(dataset)
    p = 1, 1 - noisy_rate
    for i in range(len(a)):
        s = np.random.binomial(1, p, [3, 32, 32])
        s = torch.FloatTensor(s)
        b = a[i][0]
        b = torch.FloatTensor(b)
        b.mul(s)
        b = b.numpy()
    a = tuple(a)
    return a


if __name__ == '__main__':
    main()