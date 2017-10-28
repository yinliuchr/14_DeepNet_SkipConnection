import xlrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


# table = data.sheets()[0]
#
# nrows = table.nrows
# ncols = table.ncols
#
# for i in range(ncols):
#       print table.col_values(i)
#
# tpn = table.col_values(0)
# m1_32 = table.col_values(1)
# m0_37 = table.col_values(2)
# m0_38 = table.col_values(3)
#
# plt.figure()
# # matplotlib.rc('xtick', labelsize=5)
# # matplotlib.rc('ytick', labelsize=5)
# plt.title('val_acc with training datasize')
# plt.xlabel('training datasize')
# plt.ylabel('val_acc')
# plt.plot(tpn, m1_32, 'r-x', label='dense_6_32_37748')
# plt.plot(tpn, m0_37, 'b-*', label='plain_6_37_37279')
# plt.plot(tpn, m0_38, 'g-*', label='plain_6_38_38438')
# plt.legend(loc=4)
# plt.savefig('MNIST_runs4_w32.eps')


data = xlrd.open_workbook('cnm.xlsx')


def create_dir(filename):
    if os.path.exists(filename):
        pass
    else:
        os.mkdir(filename)


table = data.sheets()[0]

# nrows = table.nrows
# ncols = table.ncols
# for i in range(ncols):
#     print table.col_values(i)

noise = np.array(table.col_values(0))
m1 = table.col_values(1) + table.col_values(2) + table.col_values(3) + table.col_values(4) + table.col_values(5)
m1 = np.array(m1).reshape(5, 10)
dense_mean = np.mean(m1, axis=0)
# dense_std = np.std(m1, axis=0)

table = data.sheets()[1]
m2 = table.col_values(1) + table.col_values(2) + table.col_values(3) + table.col_values(4) + table.col_values(5)
m2 = np.array(m2).reshape(5, 10)
plain_mean = np.mean(m2, axis=0)
# plain_std = np.std(m2, axis=0)

plt.figure()
matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)
plt.title('Validation Accuracy with Noisy Rate', fontsize=20)
plt.xlabel('noise rate', fontsize=18)
plt.ylabel('validation accuracy', fontsize=18)
#==================================================================================================================
plt.plot(noise, dense_mean, 'b-*', markersize='12', label='Dense_6_64_95956', linewidth='3')
plt.plot(noise, plain_mean, 'r-o', markersize='12', label='Plain_6_81_95843', linewidth='3')
#==================================================================================================================
plt.legend(loc=3, fontsize=13)
plt.grid()
plt.savefig('MNIST_noise_stat.eps')


# figure_plot(0, 'Validation Accuracy with Noisy Rate', 'noise rate', 'validation accuracy', 3, folder_name + '/MNIST_runs6_w32_val_acc.eps' )
# figure_plot(1, 'Validation Accuracy with Training Data Size', 'training data size', 'validation accuracy', 3, folder_name + '/MNIST_runs6_w64_val_acc.eps' )
# figure_plot(2, 'Validation Accuracy with Training Data Size', 'training data size', 'validation accuracy', 3, folder_name + '/MNIST_runs6_w128_val_acc.eps' )