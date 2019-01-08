import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


import torch
import torch.nn as nn


# from torchvision.models import resnet50

# pylint:disable=E1101

class Shift(nn.Module):
    def __init__(self, inplanes, ksize=3, dilation=1):
        super(Shift, self).__init__()
        self.dilation = dilation
        self.ksize = ksize
        self.stride = 1
        self.groups = inplanes // (ksize ** 2)
        self.shifts = []
        self.padding = (dilation * (ksize - 1)) // 2
        self.zeropad = nn.ZeroPad2d(self.padding)

        # generate random-shift for shift groups
        for _ in range(self.groups):
            dxdy = torch.randint(-(ksize - 1) // 2, (ksize + 1) // 2, (2,))
            self.shifts.append(dxdy * dilation)
        # generate zero-shift for remaining channels
        self.shifts.append(torch.randint(0, 1, (2,)))

    def forward(self, x):
        b, c, w, h = x.shape
        dk2 = self.ksize ** 2

        x = self.zeropad(x)
        accumulator = []

        # NOTE: shift channel group by group，
        for i in range(0, self.groups + 1):
            # channel indices
            start, end = i * dk2, min(c, (i + 1) * dk2)
            dx, dy = self.shifts[i]
            x1 = self.padding + dx
            y1 = self.padding + dy
            shifted_channel = x[:, start:end, y1:y1 + h, x1:x1 + w]
            accumulator.append(shifted_channel)

        output = torch.cat(accumulator, dim=1)
        return output


class CSC(nn.Module):
    def __init__(self, inplanes, planes, ksize=3, stride=1, expansion=4):
        super(CSC, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(inplanes, inplanes * expansion, 1, 1, 0),
            nn.BatchNorm2d(inplanes * expansion),
            nn.ReLU(inplace=True)
        )
        self.shift = Shift(inplanes * expansion, ksize)
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes * expansion, planes, 1, stride, 0),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        residual = x
        x = self.conv0(x)
        x = self.shift(x)
        x = self.conv1(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = x + residual
        return x


class ShiftNet(nn.Module):
    def __init__(self):
        super(ShiftNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.group1 = nn.Sequential(
            *[CSC(inplanes=32, planes=64, ksize=5, stride=1, expansion=4)] * 1,
            # *[CSC(inplanes=64, planes=64, ksize=7, stride=1, expansion=4)]*4
            nn.MaxPool2d(2, 2),
        )
        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.group1(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    x = torch.randn(3, 1, 28, 28)
    n = ShiftNet()
    import time

    start = time.time()
    # print(n)
    print(n(x).shape)
    end = time.time()
    print(end - start)
