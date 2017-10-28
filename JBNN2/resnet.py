import numpy as np
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += x
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, layers, num_classes, growth_rate):
        n = (layers - 2) / 12
        self.inplanes = growth_rate
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.relu = nn.ReLU(inplace=True)
        self.block1 = self._make_layer(growth_rate, n)
        self.block2 = self._make_layer(2 * growth_rate, n)
        self.block3 = self._make_layer(4 * growth_rate, 3 * n)
        self.block4 = self._make_layer(8 * growth_rate, n)
        self.fc = nn.Linear(8 * growth_rate * BasicBlock.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, layer_num):
        layers = [BasicBlock(self.inplanes, planes)]
        self.inplanes = planes
        for i in range(1, layer_num):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = F.avg_pool2d(x, 2)
        x = self.block2(x)
        x = F.avg_pool2d(x, 2)
        x = self.block3(x)
        x = F.avg_pool2d(x, 2)
        x = self.block4(x)

        x = F.avg_pool2d(x, 4)
        x = x.view(-1, self.inplanes)
        x = self.fc(x)

        return x

#
# def main():
#     lin_param = np.zeros([50, 8])
#     for j in range(1, 51):
#         for i in range(14, 99, 12):
#             model = ResNet(i, 10, j)
#             print('layers = ' + str(i) + ', growth_rate = ' + str(j) + ', \t\tparam_num: {}'.format(
#                 sum([p.data.nelement() for p in model.parameters()])))
#             x_ind = j - 1
#             y_ind = (i - 2) / 12 - 1
#             lin_param[x_ind, y_ind] = sum([p.data.nelement() for p in model.parameters()])
#
#     np.savetxt('linear_r.csv', lin_param, delimiter=',', fmt='%d')
#     # model = model.cuda()
#
#
# if __name__ == '__main__':
#     main()