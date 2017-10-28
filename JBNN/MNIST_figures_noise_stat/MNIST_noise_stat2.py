import xlrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


data = xlrd.open_workbook('cnm.xlsx')


def create_dir(filename):
    if os.path.exists(filename):
        pass
    else:
        os.mkdir(filename)


table = data.sheets()[0]

noise = np.array(table.col_values(0))
m1 = table.col_values(1) + table.col_values(2) + table.col_values(3) + table.col_values(4) + table.col_values(5)
m1 = np.array(m1).reshape(5, 10)
dense_mean = np.mean(m1, axis=0)
dense_std = np.std(m1, axis=0)

table = data.sheets()[1]
m2 = table.col_values(1) + table.col_values(2) + table.col_values(3) + table.col_values(4) + table.col_values(5)
m2 = np.array(m2).reshape(5, 10)
plain_mean = np.mean(m2, axis=0)
plain_std = np.std(m2, axis=0)

plt.figure()
matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)
plt.title('Validation Accuracy with Noisy Rate', fontsize=20)
plt.xlabel('noise rate', fontsize=18)
plt.ylabel('validation accuracy', fontsize=18)
#==================================================================================================================
plt.errorbar(noise, dense_mean, yerr=dense_std, label='Dense', capsize=5, elinewidth=2, markeredgewidth=2)
plt.errorbar(noise, plain_mean, yerr=plain_std, label='Plain', capsize=5, elinewidth=2, markeredgewidth=2)
#==================================================================================================================
plt.legend(loc=3, fontsize=13)
plt.grid()
plt.savefig('MNIST_noise_stat2.eps')
