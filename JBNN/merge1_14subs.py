import random
import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

tu, tb = -1.0, 1.0
epoch_num = 160
pos_bound = 15

m = 7                   # layer_num      Notice: this doesn't change the actual number of layers in the network!!!!!
r1, r2 = 4, 2           # middle_layer_node
pn1 = (m - 2) * r1 ** 2 + (m + 1) * r1 + 1
pn2 = (m - 2) * (m - 1) / 2 * r2 ** 2 + (3 * m - 3) * r2 + 2
folder_name = 'LYY_' + str(m) + '_layers-' \
              + str(pn1) + '_plain_params-'\
              + str(pn2) + '_dense_params-'


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


class NetPlain(nn.Module):
    def __init__(self):
        super(NetPlain, self).__init__()
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


class NetDense(nn.Module):
    def __init__(self):
        super(NetDense, self).__init__()
        self.linear1 = nn.Linear(1, r2)
        self.linear2 = nn.Linear(1 + r2, r2)
        self.linear3 = nn.Linear(1 + 2 * r2, r2)
        self.linear4 = nn.Linear(1 + 3 * r2, r2)
        self.linear5 = nn.Linear(1 + 4 * r2, r2)
        self.linear6 = nn.Linear(1 + 5 * r2, r2)
        self.linear7 = nn.Linear(1 + 6 * r2, 1)
        self.re = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        # # relu
        # output1 = self.re(self.linear1(x))
        # input2 = torch.cat((x, output1), 1)
        # output2 = self.re(self.linear2(input2))
        # input3 = torch.cat((input2, output2), 1)
        # output3 = self.re(self.linear3(input3))
        # input4 = torch.cat((input3, output3), 1)
        # output4 = self.re(self.linear4(input4))
        # input5 = torch.cat((input4, output4), 1)
        # output5 = self.re(self.linear5(input5))
        # input6 = torch.cat((input5, output5), 1)
        # output6 = self.re(self.linear6(input6))
        # input7 = torch.cat((input6, output6), 1)
        # output7 = self.linear7(input7)

        # tanh
        output1 = self.tanh(self.linear1(x))
        input2 = torch.cat((x, output1), 1)
        output2 = self.tanh(self.linear2(input2))
        input3 = torch.cat((input2, output2), 1)
        output3 = self.tanh(self.linear3(input3))
        input4 = torch.cat((input3, output3), 1)
        output4 = self.tanh(self.linear4(input4))
        input5 = torch.cat((input4, output4), 1)
        output5 = self.tanh(self.linear5(input5))
        input6 = torch.cat((input5, output5), 1)
        output6 = self.tanh(self.linear6(input6))
        input7 = torch.cat((input6, output6), 1)
        output7 = self.linear7(input7)

        return output7

    def output_from_each_layer(self, x):
        # # relu
        # output1 = self.re(self.linear1(x))
        # input2 = torch.cat((x, output1), 1)
        # output2 = self.re(self.linear2(input2))
        # input3 = torch.cat((input2, output2), 1)
        # output3 = self.re(self.linear3(input3))
        # input4 = torch.cat((input3, output3), 1)
        # output4 = self.re(self.linear4(input4))
        # input5 = torch.cat((input4, output4), 1)
        # output5 = self.re(self.linear5(input5))
        # input6 = torch.cat((input5, output5), 1)
        # output6 = self.re(self.linear6(input6))
        # input7 = torch.cat((input6, output6), 1)
        # output7 = self.linear7(input7)

        # tanh
        output1 = self.tanh(self.linear1(x))
        input2 = torch.cat((x, output1), 1)
        output2 = self.tanh(self.linear2(input2))
        input3 = torch.cat((input2, output2), 1)
        output3 = self.tanh(self.linear3(input3))
        input4 = torch.cat((input3, output3), 1)
        output4 = self.tanh(self.linear4(input4))
        input5 = torch.cat((input4, output4), 1)
        output5 = self.tanh(self.linear5(input5))
        input6 = torch.cat((input5, output5), 1)
        output6 = self.tanh(self.linear6(input6))
        input7 = torch.cat((input6, output6), 1)
        output7 = self.linear7(input7)

        return [output1, output2, output3, output4, output5, output6, output7]


create_dir(folder_name)


##===========================================================================================================
learning_rate = 1e-2
[x_train, y_train, x_test, y_test, x_val, y_val] = generate_dataset(30)
net = NetDense()
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
        inputs = x_train
        targets = y_train
        inputs, targets = Variable(inputs), Variable(targets)
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

x_standard = np.linspace(-pos_bound, pos_bound, 2000)
y_standard = np.sin(x_standard) / x_standard

x_test = x_test.numpy()
y_predict = y_predict.data.numpy()

y7 = y_predict_detail[6].data.numpy()

plt.figure(figsize=(150, 40))
for subplot_num in range(1, 7):
    plt.subplot(2, 7, subplot_num)
    if subplot_num == 1:
        plt.ylabel('dense', fontsize=200)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    plt.xticks([])
    plt.yticks([])
    plt.plot(x_test, y_predict_detail[subplot_num-1].data.numpy(), linewidth='6')
    plt.title(str(subplot_num), fontsize=200)

plt.subplot(2, 7, 7)
plt.axis([-pos_bound, pos_bound, -0.5, 1.0])
plt.plot(x_standard, y_standard, label='standard', linewidth='6')
plt.xticks([])
plt.yticks([])
plt.plot(x_test, y7, label='y7', linewidth='6')
# plt.legend()
plt.title('7', fontsize=200)

##===========================================================================================================
learning_rate = 1e-2
[x_train, y_train, x_test, y_test, x_val, y_val] = generate_dataset(400)
net = NetPlain()
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
        inputs = x_train
        targets = y_train
        inputs, targets = Variable(inputs), Variable(targets)
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

x_standard = np.linspace(-pos_bound, pos_bound, 2000)
y_standard = np.sin(x_standard) / x_standard

x_test = x_test.numpy()
y_predict = y_predict.data.numpy()
y7 = y_predict_detail[6].data.numpy()


for subplot_num in range(1, 7):
    plt.subplot(2, 7, subplot_num+7)
    if subplot_num == 1:
        plt.ylabel('plain', fontsize=200)
    plt.axis([-pos_bound, pos_bound, tu, tb])
    plt.xticks([])
    plt.yticks([])
    plt.plot(x_test, y_predict_detail[subplot_num-1].data.numpy(), linewidth='6')
    plt.title(str(subplot_num), fontsize=200)

plt.subplot(2, 7, 14)
plt.axis([-pos_bound, pos_bound, -0.5, 1.0])
plt.plot(x_standard, y_standard, label='standard', linewidth='6')
plt.xticks([])
plt.yticks([])
plt.plot(x_test, y7, label='y7', linewidth='6')
# plt.legend()
plt.title('7', fontsize=200)

fig_title = folder_name + '/Comparison_14sub.eps'

plt.savefig(fig_title)
