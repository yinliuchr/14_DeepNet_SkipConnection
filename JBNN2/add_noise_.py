# a = []
# b = [1,2,3]
# a.append(b)
# print a

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
import torchvision.datasets as datasets
from matplotlib import transforms

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

transform_train = transforms.Compose([transforms.ToTensor(), normalize])
# transform_train = transforms.Compose([transforms.ToTensor()])
# print np.shape(a)
# print len(a)

# def add_noise(dataset, noisy_rate):
#     """Add noise to dataset
#     :param noisy_rate: int, choose from [1, 2, 3, ..., 1024]
#     """
#     a = list(dataset)
#     for i in range(len(a)):
#         for j in range(noisy_rate):
#             r = [random.randint(0, 31) for m in range(6)]
#             a[i][0][0, r[0], r[1]] = 0
#             a[i][0][1, r[2], r[3]] = 0
#             a[i][0][2, r[4], r[5]] = 0
#     a = tuple(a)
#     return a


def add_noise(dataset, noisy_rate):
    a = list(dataset)
    n, p = 1, 1 - noisy_rate
    for i in range(len(a)):
        s = np.random.binomial(n, p, [3, 32, 32])
        s = torch.FloatTensor(s)
        b = a[i][0]
        b = torch.FloatTensor(b)
        b.mul_(s)
    a = tuple(a)
    return a

a = datasets.CIFAR10('../data', train=True, download=True, transform=transform_train)

jbn = add_noise(a, 0.1)
print jbn

