import xlrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


data = xlrd.open_workbook('Cifar100_runs_tpn2.xlsx')

def figure_plot(sheetes_num, title, x_name, y_name, loc, fig_name):
    table = data.sheets()[sheetes_num]

    nrows = table.nrows
    ncols = table.ncols

    for i in range(ncols):
        print table.col_values(i)

    tpn = table.col_values(0)
    m_d = table.col_values(1)
    m_p = table.col_values(2)
    m_r = table.col_values(3)

    plt.figure()
    matplotlib.rc('xtick', labelsize=13)
    matplotlib.rc('ytick', labelsize=13)
    plt.title(title, fontsize=20)
    plt.xlabel(x_name, fontsize=18)
    plt.ylabel(y_name, fontsize=18)
    #==================================================================================================================
    plt.plot(tpn, m_d, 'b-*', markersize='12', label='Dense_46_18671', linewidth='3')
    plt.plot(tpn, m_p, 'r-o', markersize='12', label='Plain_44_2546442', linewidth='3')
    plt.plot(tpn, m_r, 'g-h', markersize='12', label='Res_44_130293', linewidth='3')

    # plt.plot(tpn, m1, 'b-*', markersize='12', label='Dense_6_64_95956', linewidth='3')
    # plt.plot(tpn, m0, 'r-o', markersize='12', label='Plain_6_81_95843', linewidth='3')

    # plt.plot(tpn, m1, 'b-*', markersize='12', label='Dense_6_128_273812', linewidth='3')
    # plt.plot(tpn, m0, 'r-o', markersize='12', label='Plain_6_175_272845', linewidth='3')
    #==================================================================================================================
    plt.legend(loc=loc, fontsize = 18)
    plt.grid()
    plt.savefig(fig_name)


def create_dir(filename):
    if os.path.exists(filename):
        pass
    else:
        os.mkdir(filename)


folder_name = 'Cifar100_figures_tpn'
# create_dir(folder_name)

figure_plot(0, 'Absolute Validation Accuracy with Training Data Size',
            'training data size', 'validation accuracy', 4,
            folder_name + '/Cifar100_runs_abs_acc2.eps' )
