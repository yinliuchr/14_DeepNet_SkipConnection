import xlrd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os


data = xlrd.open_workbook('cnm.xlsx')

table = data.sheets()[0]

# nrows = table.nrows
# ncols = table.ncols
#
# for i in range(ncols):
#     print table.col_values(i)

tpn = np.array(table.col_values(0))
m1 = table.col_values(1) + table.col_values(2) + table.col_values(3) + table.col_values(4) + table.col_values(5)
m1 = np.array(m1).reshape(5, 10)
dense_mean = np.mean(m1, axis=0)
dense_std = np.std(m1, axis=0)

plt.figure()
matplotlib.rc('xtick', labelsize=13)
matplotlib.rc('ytick', labelsize=13)

plt.errorbar(tpn, dense_mean, yerr=dense_std, label='dense', capsize=5, elinewidth=2, markeredgewidth=2)
plt.legend()

plt.xlabel('noise', fontsize=18)
plt.ylabel('val_acc', fontsize=18)
plt.title('K', fontsize=20)

plt.savefig('L.eps')

print type(m1)
print m1

