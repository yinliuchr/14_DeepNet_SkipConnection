# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt

## example data
# x = np.arange(0.1, 4, 0.5)
# y = np.exp(-x)
#
# # example variable error bar values
# yerr = 0.1 + 0.2*np.sqrt(x)
# xerr = 0.1 + yerr

# First illustrate basic pyplot interface, using defaults where possible.
# plt.figure()
#
# plt.errorbar(x, y, yerr=0.4)
# plt.ylim(0.1,0.2)
# plt.title("Simplest errorbars, 0.2 in x, 0.4 in y")
# plt.savefig('k.eps')
#
# x = np.arange(40)
# # print x
# x = x.reshape(5,8)
# print x
# s = np.mean(x, axis=0)
# t = np.std(x, axis=0)
# print s
# print t
# print np.std(np.array([20, 28, 36, 44, 52]))

# mu, sigma = 0, 0.1 # mean and standard deviation
# s = np.random.normal(mu, sigma)
# print s
# # print np.mean(s)
# print np.std(s)

# Make some fake data.
# from matplotlib.ticker import FormatStrFormatter
#
# a = b = np.arange(0, 3, 0.2)
# c = np.exp(a)
# d = c[::-1]
# matplotlib.rc('xtick', labelsize=5)
# matplotlib.rc('ytick', labelsize=5)
# # Create plots with pre-defined labels.
# plt.plot(a, c, label='Model length')
# plt.scatter(a, d, label='Data length', marker='.')
# plt.plot(a, c + d, label='Total message length')
#
# plt.legend(prop={'size':1})
#
# # Put a nicer background color on the legend.
# # legend.get_frame().set_facecolor('#00FFCC')
#
# plt.savefig('j.eps')

# pr_dict = {3:4, 4:4, 5:5, 6:5, 7:6, 8:6, 9:6, 10:7, 11:7, 12:7, 13:8,
#                14:8, 15:8, 16:8, 17:8, 18:9, 19:9, 20:9, 21:9, 22:10, 23:10, 24:10}
# print pr_dict[9]
# print type(pr_dict[9])

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

tu, tb = -0.1, 1.0
epoch_num = 80
pos_bound = 15
global_lr = 1e-2

tpn = 60
# m = 7                   # layer_num      Notice: this doesn't change the actual number of layers in the network!!!!!
# r1, r2 = 4, 2           # middle_layer_node
# pn1 = (m - 2) * r1 ** 2 + (m + 1) * r1 + 1
# pn2 = (m - 2) * (m - 1) / 2 * r2 ** 2 + (3 * m - 3) * r2 + 2
# folder_name =  str(m) + '_layers-' \
#               + str(pn1) + '_plain_params-'\
#               + str(pn2) + '_dense_params-'

folder_name = 'Vdn_to_24_stat7--' + str(tpn) + '_tpn'


def fitting_func(x):
    return math.sin(x) / x


def create_dir(filename):
    if os.path.exists(filename):
        pass
    else:
        os.mkdir(filename)


def generate_dataset(tpn):
    a = tpn
    liu = []
    for i in range(a):
        # t1 = random.uniform(-pos_bound, pos_bound)
        t1 = -pos_bound + 2.0 * pos_bound / a * (i + 0.5)
        t2 = fitting_func(t1) + np.random.normal(0, 0.05)
        liu.append(t1)
        liu.append(t2)



        # if data.any():
        #     data = np.vstack((data, [[t1, t2]]))
        # else:
        #     data = np.array([[t1, t2]])

    data = np.array(liu)
    data = data.reshape(a, 2)
    data_arg = np.argsort(data[:, 0])
    data = data[data_arg]
    x_train = data[:, 0]
    y_train = data[:, 1]

    # b = raw_input('number of testing_points:')
    # b = int(b)
    b = 1000
    liu = []
    for i in range(b):
        t1 = random.uniform(-pos_bound, pos_bound)
        t2 = fitting_func(t1)
        liu.append(t1)
        liu.append(t2)
    data = np.array(liu)
    data = data.reshape(b, 2)
    data_arg = np.argsort(data[:, 0])
    data = data[data_arg]
    x_test = data[:, 0]
    y_test = data[:, 1]

    c = 400
    liu = []
    for i in range(c):
        t1 = random.uniform(-pos_bound, pos_bound)
        t2 = fitting_func(t1)
        liu.append(t1)
        liu.append(t2)
    data = np.array(liu)
    data = data.reshape(c, 2)
    data_arg = np.argsort(data[:, 0])
    data = data[data_arg]
    x_val = data[:, 0]
    y_val = data[:, 1]

    x_train = torch.FloatTensor([[i] for i in x_train])
    y_train = torch.FloatTensor([[i] for i in y_train])
    x_test = torch.FloatTensor([[i] for i in x_test])
    y_test = torch.FloatTensor([[i] for i in y_test])
    x_val = torch.FloatTensor([[i] for i in x_val])
    y_val = torch.FloatTensor([[i] for i in y_val])
    return [x_train, y_train, x_test, y_test, x_val, y_val]


class NetPlain(nn.Module):
    def __init__(self, layers, r1):
        super(NetPlain, self).__init__()
        self.layers = layers
        self.r1 = r1
        self.param_num = (layers - 2) * r1 ** 2 + (layers + 1) * r1 + 1
        linear = [nn.Linear(1, r1)]
        for i in range(layers - 2):
            linear.append(nn.Linear(r1, r1))
        linear.append(nn.Linear(r1, 1))
        self.linear = nn.Sequential(*linear)
        self.re = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = [self.re(self.linear[0](x))]
        for i in range(self.layers - 2):
            output.append(self.re(self.linear[i + 1](output[-1])))
        output.append(self.linear[self.layers - 1](output[-1]))

        return output[-1]

    def output_from_each_layer(self, x):
        output = [self.re(self.linear[0](x))]
        for i in range(self.layers - 2):
            output.append(self.re(self.linear[i + 1](output[-1])))
        output.append(self.linear[self.layers - 1](output[-1]))

        cop = []
        for i in range(len(output)):
            cop.append(output[i][:, -3:])
        return cop


class NetRes(nn.Module):
    def __init__(self, layers, r2):
        super(NetRes, self).__init__()
        self.layers = layers
        self.r2 = r2
        self.param_num = (layers - 2) * r2 ** 2 + (layers + 1) * r2 + 1
        linear = [nn.Linear(1, r2)]
        for i in range(layers - 2):
            linear.append(nn.Linear(r2, r2))
        linear.append(nn.Linear(r2, 1))
        self.linear = nn.Sequential(*linear)
        self.re = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = [self.re(self.linear[0](x))]
        for i in range(self.layers - 2):
            output.append(self.re(self.linear[i + 1](output[-1]) + output[-1]))
        output.append(self.linear[self.layers - 1](output[-1]))

        return output[-1]

    def output_from_each_layer(self, x):
        output = [self.re(self.linear[0](x))]
        for i in range(self.layers - 2):
            output.append(self.re(self.linear[i + 1](output[-1]) + output[-1]))
        output.append(self.linear[self.layers - 1](output[-1]))

        cop = []
        for i in range(len(output)):
            cop.append(output[i][:, -3:])
        return cop


class NetDense(nn.Module):
    def __init__(self, layers, r3):
        super(NetDense, self).__init__()
        self.layers = layers
        self.r3 = r3
        self.param_num = (layers - 2) * (layers - 1) / 2 * r3 ** 2 + (3 * layers - 3) * r3 + 2
        linear = [nn.Linear(1, r3)]
        for i in range(layers - 2):
            linear.append(nn.Linear(1 + (i + 1) * r3, r3))
        linear.append(nn.Linear(1 + (layers - 1) * r3, 1))
        self.linear = nn.Sequential(*linear)
        # self.linear = linear[:]
        self.re = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        output = [self.re(self.linear[0](x))]
        input = [torch.cat((x, output[-1]), 1)]
        for i in range(self.layers - 2):
            output.append(self.re(self.linear[i + 1](input[-1])))
            input.append(torch.cat((input[-1], output[-1]), 1))
        output.append(self.linear[self.layers - 1](input[-1]))

        return output[-1]

    def output_from_each_layer(self, x):
        output = [self.re(self.linear[0](x))]
        input = [torch.cat((x, output[-1]), 1)]
        for i in range(self.layers - 2):
            output.append(self.re(self.linear[i + 1](input[-1])))
            input.append(torch.cat((input[-1], output[-1]), 1))
        output.append(self.linear[self.layers - 1](input[-1]))

        cop = []
        for i in range(len(output)):
            cop.append(output[i][:, -3:])
        return cop


def jbn(learning_rate, layers, r, net_choice):
    [x_train, y_train, x_test, y_test, x_val, y_val] = generate_dataset(tpn)
    if net_choice == 1:
        net = NetPlain(layers, r)
    elif net_choice ==2:
        net = NetRes(layers, r)
    else:
        net = NetDense(layers, r)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    def adjust_learning_rate(optimizer, epoch):
        lr = learning_rate * (0.1 ** (epoch // (epoch_num * 0.5))) * (0.1 ** (epoch // (epoch_num * 0.75)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    loss_list = []
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        adjust_learning_rate(optimizer, epoch)
        for i in range(400):
            # get the inputs
            inputs = x_train
            targets = y_train
            # wrap them in Variable
            inputs, targets = Variable(inputs), Variable(targets)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        input_val, target_val = Variable(x_val), Variable(y_val)
        output_val = net(input_val)
        loss_val = criterion(output_val, target_val)
        loss_list.append(loss_val.data[0])
        print('========================== >>>>>> val_loss is: %.4f' % (loss_val.data[0]))
        print '===================================learning_rate = ' + str(optimizer.param_groups[0]['lr'])

    print('Finished Training')
    print('============End of training -- learning rate = ' + str(learning_rate) + '\n\n\n')

    x_standard = np.linspace(-pos_bound, pos_bound, 2000)
    y_standard = np.sin(x_standard) / x_standard

    y_predict_detail = net.output_from_each_layer(Variable(x_test))
    x_test = x_test.numpy()

    y = []
    for i in range(layers):
        y.append(y_predict_detail[i].data.numpy())

    plt.figure()
    matplotlib.rc('xtick', labelsize=5)
    matplotlib.rc('ytick', labelsize=5)
    for i in range(layers - 1):
        plt.subplot(5, 5, i + 1)
        plt.axis([-pos_bound, pos_bound, tu, tb])
        plt.plot(x_test, y[i])

    plt.subplot(5, 5, layers)
    plt.axis([-pos_bound, pos_bound, -0.5, 1.0])
    plt.plot(x_standard, y_standard, label='standard')
    plt.scatter(x_train.numpy(), y_train.numpy(), label='training_data', marker='.')
    plt.plot(x_test, y[-1], label='test')
    plt.legend(prop={'size':1})

    plt.subplot(5, 5, 25)
    plt.plot(np.arange(1, epoch_num + 1), np.array(loss_list))

    fig_title = folder_name \
                + '/' \
                + str(net_choice) \
                + '--' \
                + str(layers) + '_layers--' \
                + str(net.param_num) + '_params' \
                + '.eps'
    plt.savefig(fig_title)
    return loss_list[-1]


def main():
    create_dir(folder_name)
    x_axis = np.arange(3, 25)  ### layers
    flp, flr, fld = [], [], []
    # pr_dict = {3:4, 4:4, 5:5, 6:5, 7:6, 8:6, 9:6, 10:7, 11:7, 12:7, 13:8,
    #            14:8, 15:8, 16:8, 17:8, 18:9, 19:9, 20:9, 21:9, 22:10, 23:10, 24:10}
    pr_dict = {3: 6, 4: 7, 5: 7, 6: 8, 7: 9, 8: 10, 9: 10, 10: 11, 11: 11, 12: 12, 13: 12,
               14: 13, 15: 13, 16: 14, 17: 14, 18: 15, 19: 15, 20: 15, 21: 16, 22: 16, 23: 17, 24: 17}
    for j in range(10):
        for i in x_axis:
            flp.append(jbn(global_lr, i, pr_dict[i], 1))
            flr.append(jbn(global_lr, i, pr_dict[i], 2))
            fld.append(jbn(global_lr, i, 5, 3))

    flp = np.array(flp).reshape(10, 22)
    flr = np.array(flr).reshape(10, 22)
    fld = np.array(fld).reshape(10, 22)

    print flp
    print flr
    print fld

    plain_mean = np.mean(flp, axis=0)
    plain_std = np.std(flp, axis=0)

    res_mean = np.mean(flr, axis=0)
    res_std = np.std(flr, axis=0)

    dense_mean = np.mean(fld, axis=0)
    dense_std = np.std(fld, axis=0)

    plt.figure()
    plt.ylim(0, 0.04)
    plt.errorbar(x_axis, plain_mean, yerr=plain_std, label='plain')
    plt.errorbar(x_axis, res_mean, yerr=res_std, label='res')
    plt.errorbar(x_axis, dense_mean, yerr=dense_std, label='dense')
    plt.legend()
    plt.title('loss_vs_depth--noisy_stat')
    plt.savefig(folder_name + '/loss_vs_depth--noisy_stat.eps')


if __name__ == '__main__':
    main()
