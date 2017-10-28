import random
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


tpn = 60
folder_name = 'Vdn_to_24_stat_X--' + str(tpn) + '_tpn'

##=========================================================================================================
x_axis = np.load('x_axis.npy')

plain_mean = np.load('plain_mean.npy')
res_mean = np.load('res_mean.npy')
dense_mean = np.load('dense_mean.npy')

plain_std = np.load('plain_std.npy')
res_std = np.load('res_std.npy')
dense_std = np.load('dense_std.npy')
##=========================================================================================================
# print x_axis
# print type(x_axis)
x_axis2 = np.arange(3, 15)

plt.figure()
plt.ylim(0, 0.01)
matplotlib.rc('xtick', labelsize=200)
matplotlib.rc('ytick', labelsize=200)
plt.errorbar(x_axis2, plain_mean[:12], yerr=plain_std[:12], label='Plain')
plt.errorbar(x_axis, res_mean, yerr=res_std, label='Res')
plt.errorbar(x_axis, dense_mean, yerr=dense_std, label='Dense')
plt.legend()
plt.xlabel('depth', fontsize=18)
plt.ylabel('validation loss', fontsize=18)
plt.title('Loss with Depth--Statistical', fontsize=20)
plt.savefig(folder_name + '/loss_vs_depth--noisy_stat4.eps')