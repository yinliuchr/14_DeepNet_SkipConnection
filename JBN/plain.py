import random
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

tu, tb = -1.0, 1.0
epoch_num = 200
pos_bound = 15

tpn = 50
m = 7  # layer_num      Notice: this doesn't change the actual number of layers in the network!!!!!
r1 = 12  # middle_layer_node
param_num = (m - 2) * r1 ** 2 + (m + 1) * r1 + 1
folder_name = 'plain-' \
              + str(tpn) + '_points-' \
              + str(m) + '_layers-' \
              + str(param_num) + '_params'


def fitting_func(x):
    return math.sin(x) / x


def create_dir(filename):
    if os.path.exists(filename):
        pass
    else:
        os.mkdir(filename)


def generate_dataset():
    a = tpn
    liu = []
    for i in range(a):
        t1 = random.uniform(-pos_bound, pos_bound)
        t2 = fitting_func(t1)
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


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # linear = [nn.Linear(1, r1)]
        # for i in range(w1 - 2):
        #     linear.append(nn.Linear(r1, r1))
        # linear.append(nn.Linear(r1, 1))
        # self.linear = linear
        #
        self.linear1 = nn.Linear(1, r1)
        self.linear2 = nn.Linear(r1, r1)
        self.linear3 = nn.Linear(r1, r1)
        self.linear4 = nn.Linear(r1, r1)
        self.linear5 = nn.Linear(r1, r1)
        self.linear6 = nn.Linear(r1, r1)
        self.linear7 = nn.Linear(r1, 1)
        self.re = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # # relu
        # output1 = self.re(self.linear1(x))
        # output2 = self.re(self.linear2(output1))
        # output3 = self.re(self.linear3(output2))
        # output4 = self.re(self.linear4(output3))
        # output5 = self.re(self.linear5(output4))
        # output6 = self.re(self.linear6(output5))
        # output7 = self.linear7(output6)


        # tanh
        output1 = self.tanh(self.linear1(x))
        output2 = self.tanh(self.linear2(output1))
        output3 = self.tanh(self.linear3(output2))
        output4 = self.tanh(self.linear4(output3))
        output5 = self.tanh(self.linear5(output4))
        output6 = self.tanh(self.linear6(output5))
        output7 = self.linear7(output6)

        return output7

    def output_from_each_layer(self, x):
        # # relu
        # output1 = self.re(self.linear1(x))
        # output2 = self.re(self.linear2(output1))
        # output3 = self.re(self.linear3(output2))
        # output4 = self.re(self.linear4(output3))
        # output5 = self.re(self.linear5(output4))
        # output6 = self.re(self.linear6(output5))
        # output7 = self.linear7(output6)



        # tanh
        output1 = self.tanh(self.linear1(x))
        output2 = self.tanh(self.linear2(output1))
        output3 = self.tanh(self.linear3(output2))
        output4 = self.tanh(self.linear4(output3))
        output5 = self.tanh(self.linear5(output4))
        output6 = self.tanh(self.linear6(output5))
        output7 = self.linear7(output6)

        return [output1, output2, output3, output4, output5, output6, output7]
        # return [output1, output2, output3, output4]


def jbn(learning_rate):
    [x_train, y_train, x_test, y_test, x_val, y_val] = generate_dataset()

    net = Net()
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

    y_predict = net(Variable(x_test))

    y_predict_detail = net.output_from_each_layer(Variable(x_test))

    # input_num1 = input1.data.numpy()
    # output_num1 = output1.data.numpy()
    # output_num2 = output2.data.numpy()
    # output_num3 = output3.data.numpy()
    # output_num4 = output4.data.numpy()
    # # dict1 = {}
    # # for i in range(a):
    # #     dict1[input_num1[i]] = output_num1[i]
    #
    #
    # plt.plot(input_num1, output_num1)
    # plt.savefig('jbn.eps')
    # plt.show()

    # x_train = x_train.numpy()
    # y_train = y_train.numpy()
    x_standard = np.linspace(-pos_bound, pos_bound, 2000)
    y_standard = np.sin(x_standard) / x_standard

    x_test = x_test.numpy()
    y_predict = y_predict.data.numpy()
    y1 = y_predict_detail[0].data.numpy()
    y2 = y_predict_detail[1].data.numpy()
    y3 = y_predict_detail[2].data.numpy()
    y4 = y_predict_detail[3].data.numpy()
    y5 = y_predict_detail[4].data.numpy()
    y6 = y_predict_detail[5].data.numpy()
    y7 = y_predict_detail[6].data.numpy()
    # print type(y1)

    # plt.plot(x_train, y_train)

    ### figure 1
    # plt.figure(1)
    # plt.plot(x_standard, y_standard, label='standard')
    # plt.plot(x_test, y_predict, label='learned')
    # plt.legend()
    # plt.title('Standard and Learned')
    # plt.savefig('densenet_1000_7_tanh.eps')

    ### figure 2
    plt.figure()

    plt.subplot(3, 3, 1)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    # plt.plot(x_standard, y_standard, label='standard')
    plt.plot(x_test, y1, label='y1')
    # plt.legend()
    # plt.title('1')

    plt.subplot(3, 3, 2)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    # plt.plot(x_standard, y_standard, label='standard')
    plt.plot(x_test, y2, label='y2')
    # plt.legend()
    # plt.title('2')

    plt.subplot(3, 3, 3)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    # plt.plot(x_standard, y_standard, label='standard')
    plt.plot(x_test, y3, label='y3')
    # plt.legend()
    # plt.title('3')

    plt.subplot(3, 3, 4)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    # plt.plot(x_standard, y_standard, label='standard')
    plt.plot(x_test, y4, label='y4')
    # plt.legend()
    # plt.title('4')

    plt.subplot(3, 3, 5)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    # plt.plot(x_standard, y_standard, label='standard')
    plt.plot(x_test, y5, label='y5')
    # plt.legend()
    # plt.title('5')

    plt.subplot(3, 3, 6)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    # plt.plot(x_standard, y_standard, label='standard')
    plt.plot(x_test, y6, label='y6')
    # plt.legend()
    # plt.title('6')

    plt.subplot(3, 3, 7)
    plt.axis([-pos_bound, pos_bound, -0.5, 1.0])
    plt.plot(x_standard, y_standard, label='standard')
    plt.plot(x_test, y7, label='y7')
    # plt.legend()
    # plt.title('7')

    plt.subplot(3, 3, 8)
    # plt.axis([-pos_bound, pos_bound, tu, tb])
    # plt.plot(x_standard, y_standard, label='standard')
    # plt.plot(x_test, y7, label='y7')
    plt.plot(np.arange(1, epoch_num + 1), np.array(loss_list))

    fig_title = folder_name + '/plain_' + str(learning_rate) + '.eps'
    plt.savefig(fig_title)
    return loss_list[-1]


def main():
    create_dir(folder_name)
    lr_list = [8e-3, 1e-2, 2e-2, 3e-2, 4e-2, 6e-2, 8e-2]
    x_axis = np.log10(lr_list)
    final_loss = []
    min_loss, min_loss_lr = 1, 1
    for lr in lr_list:
        temp = jbn(lr)
        if temp < min_loss:
            min_loss = temp
            min_loss_lr = lr
        final_loss.append(temp)
    plt.figure()
    plt.plot(np.array(x_axis), np.array(final_loss))
    plt.title('loss_vs_lr')
    plt.savefig(folder_name + '/plain__loss_vs_lr.eps')
    print
    print 'r1 = ' + str(r1)
    print 'Min_loss = ' + str(min_loss)
    print 'Min_loss_lr = ' + str(min_loss_lr)

if __name__ == '__main__':
    main()