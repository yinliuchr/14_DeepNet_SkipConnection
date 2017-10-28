import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckLayer(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckLayer, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, in_planes, growth_rate, nb_layers, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(in_planes, growth_rate, nb_layers, dropRate)

    def _make_layer(self, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(BottleneckLayer(in_planes + i * growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate, reduction=0.5, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 6

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = DenseBlock(in_planes, growth_rate, n, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionLayer(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))

        # 2nd Layer
        self.block2 = DenseBlock(in_planes, growth_rate, n, dropRate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionLayer(in_planes, int(math.floor(in_planes * reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes * reduction))

        # 3rd block
        self.block3 = DenseBlock(in_planes, growth_rate, n, dropRate)
        in_planes = int(in_planes + n * growth_rate)

        # global average pooling and classifier
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
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


# def main():
#     lin_param = np.zeros([40, 16])
#     for j in range(1, 41):
#         for i in range(10, 101, 6):
#             model = DenseNet3(i, 10, j, reduction=0.5, dropRate=0)
#             print('layers = ' + str(i) + ', growth_rate = ' + str(j) + ', \tparam_num: {}'.format(
#                 sum([p.data.nelement() for p in model.parameters()])))
#             x_ind = j - 1
#             y_ind = (i - 4) / 6 - 1
#             lin_param[x_ind, y_ind] = sum([p.data.nelement() for p in model.parameters()])
#
#     np.savetxt('linear_d.csv', lin_param, delimiter=',', fmt='%d')
#     # model = model.cuda()
#
#
# if __name__ == '__main__':
#     main()
