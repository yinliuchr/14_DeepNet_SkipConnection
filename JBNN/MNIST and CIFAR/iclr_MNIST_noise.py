import xlrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


data = xlrd.open_workbook('MNIST_runs6_noise.xlsx')

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


def figure_plot(sheetes_num, title, x_name, y_name, loc, fig_name):
    table = data.sheets()[sheetes_num]

    nrows = table.nrows
    ncols = table.ncols

    for i in range(ncols):
        print table.col_values(i)

    tpn = table.col_values(0)
    m1 = table.col_values(1)
    m0 = table.col_values(2)

    plt.figure()
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    plt.title(title, fontsize=20)
    plt.xlabel(x_name, fontsize=18)
    plt.ylabel(y_name, fontsize=18)
    #==================================================================================================================
    plt.plot(tpn, m1, 'b-*', markersize='12', label='Dense_6_32_37748', linewidth='3')
    plt.plot(tpn, m0, 'r-o', markersize='12', label='Plain_6_37_37279', linewidth='3')
    m0_38 = table.col_values(3)
    plt.plot(tpn, m0_38, 'g-o', markersize='12', label='Plain_6_38_38438', linewidth='3')

    # plt.plot(tpn, m1, 'b-*', markersize='12', label='Dense_6_64_95956', linewidth='3')
    # plt.plot(tpn, m0, 'r-o', markersize='12', label='Plain_6_81_95843', linewidth='3')

    # plt.plot(tpn, m1, 'b-*', markersize='12', label='Dense_6_128_273812', linewidth='3')
    # plt.plot(tpn, m0, 'r-o', markersize='12', label='Plain_6_175_272845', linewidth='3')
    #==================================================================================================================
    plt.legend(loc=loc, fontsize=13)
    plt.grid()
    plt.savefig(fig_name)


def create_dir(filename):
    if os.path.exists(filename):
        pass
    else:
        os.mkdir(filename)


folder_name = 'MNIST_figures_noise'
create_dir(folder_name)

figure_plot(0, 'Validation Accuracy with Noisy Rate', 'noise rate', 'validation accuracy', 3, folder_name + '/MNIST_runs6_w32_val_acc.eps' )
# figure_plot(1, 'Validation Accuracy with Training Data Size', 'training data size', 'validation accuracy', 3, folder_name + '/MNIST_runs6_w64_val_acc.eps' )
# figure_plot(2, 'Validation Accuracy with Training Data Size', 'training data size', 'validation accuracy', 3, folder_name + '/MNIST_runs6_w128_val_acc.eps' )