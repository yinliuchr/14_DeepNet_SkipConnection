import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlainLayer(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(PlainLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class PlainBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, nb_layers, dropRate=0.0):
        super(PlainBlock, self).__init__()
        self.layer = self._make_layer(in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, in_planes, growth_rate, nb_layers, dropRate):
        layers = [PlainLayer(in_planes, growth_rate, dropRate)]
        for i in range(nb_layers - 1):
            layers.append(PlainLayer(growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class PlainNet(nn.Module):
    def __init__(self, depth, num_classes, growth_rate, dropRate=0.0):
        super(PlainNet, self).__init__()
        n = (depth - 2) / 6
        self.conv1 = nn.Conv2d(3, growth_rate, kernel_size=3, stride=1,padding=1, bias=False)
        self.block1 = PlainBlock(growth_rate, growth_rate, n, dropRate)
        self.block2 = PlainBlock(growth_rate, 2 * growth_rate, n, dropRate)
        self.block3 = PlainBlock(2 * growth_rate, 4 * growth_rate, 3 * n, dropRate)
        self.block4 = PlainBlock(4 * growth_rate, 8 * growth_rate, n, dropRate)
        in_planes = 8 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = F.avg_pool2d(out, 2)
        out = self.block2(out)
        out = F.avg_pool2d(out, 2)
        out = self.block3(out)
        out = F.avg_pool2d(out, 2)
        out = self.block4(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(-1, self.in_planes)
        return self.fc(out)

#
# def main():
#     lin_param = np.zeros([50, 16])
#     for j in range(1, 51):
#         for i in range(8, 99, 6):
#             model = PlainNet(i, 10, j, dropRate=0)
#             print('layers = ' + str(i) + ', growth_rate = ' + str(j) + ', \t\tparam_num: {}'.format(
#                 sum([p.data.nelement() for p in model.parameters()])))
#             x_ind = j - 1
#             y_ind = (i - 2) / 6 - 1
#             lin_param[x_ind, y_ind] = sum([p.data.nelement() for p in model.parameters()])
#
#     np.savetxt('linear_p.csv', lin_param, delimiter=',', fmt='%d')
#     # model = model.cuda()
#
#
# if __name__ == '__main__':
#     main()