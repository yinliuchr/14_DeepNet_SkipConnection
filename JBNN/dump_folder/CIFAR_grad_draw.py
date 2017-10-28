import xlrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


data = xlrd.open_workbook('hjh.xlsx')

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


def figure_plot(sheetes_num, title, x_name, y_name, loc, fig_name, col):
    table = data.sheets()[sheetes_num]

    nrows = table.nrows
    ncols = table.ncols

    for i in range(ncols):
        print table.col_values(i)

    tpn = table.col_values(0)
    m = table.col_values(col)
    # m = table.col_values(2)

    plt.figure()
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    plt.title(title, fontsize=20)
    plt.xlabel(x_name, fontsize=18)
    plt.ylabel(y_name, fontsize=18)
    #==================================================================================================================
    plt.plot(tpn, m, 'b-*', markersize='12', linewidth='3')
    #==================================================================================================================
    plt.legend(loc=loc, fontsize = 18)
    plt.grid()
    plt.savefig(fig_name)


figure_plot(0, 'Validation Accuracy with Number of Skip Connections',
            'skip connections number', 'validation accuracy', 4,
            'Cifar100_1W_SC.eps', 1 )

figure_plot(0, 'Validation Accuracy with Number of Skip Connections',
            'skip connections number', 'validation accuracy', 4,
            'Cifar100_5W_SC.eps', 2)