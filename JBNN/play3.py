import xlrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


plt.figure()
plt.ylim(0,2)
x_axis = np.arange(5)
pm = np.array([3, 2, 4, 5, 4])
ps = np.array([2, 1, 2, 1, 2])
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
plt.errorbar(x_axis, pm, yerr=ps, label='Plain')
plt.legend()
plt.xlabel('depth', fontsize=18)
plt.ylabel('validation loss', fontsize=18)
plt.title('Loss with Depth--Statistical', fontsize=20)
plt.savefig('LL.eps')