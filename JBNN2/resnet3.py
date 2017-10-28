import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,in_planes,num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = in_planes
        self.planes = in_planes
        self.conv1 = conv3x3(3,in_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        #print(num_blocks[0])
        self.layer1 = self._make_layer(BasicBlock, self.in_planes, num_blocks[0], stride=1)
        
        self.layer2 = self._make_layer(BasicBlock, 2 * (self.planes), num_blocks[1], stride=2)
        
        self.layer3 = self._make_layer(BasicBlock, 4 * (self.planes), num_blocks[2], stride=2)
        #print(self.planes)        
        self.linear = nn.Linear(4*(self.planes)*BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        #print(num_blocks)
        #print(planes)
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
#         out = self.layer4(out)
        out = F.avg_pool2d(out,8)
        #print(out.size())
        out = out.view(out.size(0), 4*self.planes)
        out = self.linear(out)
        
        return out






# def main():
#     lin_param = np.zeros([50, 16])
#     for j in range(1, 51):
#         for i in range(8, 99, 6):
#             model = ResNet(i, 10, j, dropRate=0)
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