import torch
from torch import nn
from collections import OrderedDict


class conv_block(nn.Module):  # 标准卷积模块
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.active = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.active(x)
        return x


class depthwise_conv_block(nn.Module):  # 深度可分离卷积块
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=ch_in, bias=False)
        self.bn1 = nn.BatchNorm2d(ch_in)
        self.active = nn.ReLU6()
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x = self.depth_conv(x)  # 逐通道卷积
        x = self.bn1(x)
        x = self.active(x)
        x = self.point_conv(x)  # 逐点卷积
        x = self.bn2(x)
        x = self.active(x)
        return x


class SimpleMobileNet(nn.Module):  # 基于 MobileNetV1 简化的神经网络
    def __init__(self, classes, ch_in, dropuout_rate=0.0):
        super().__init__()
        self.conv_block1 = conv_block(ch_in, 32)
        self.depthwise_conv_block = nn.Sequential(OrderedDict([
            ('dpethwise_conv1', depthwise_conv_block(32, 64)),
            ('dpethwise_conv2', depthwise_conv_block(64, 128)),
            ('dpethwise_conv3', depthwise_conv_block(128, 256))
        ]))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=dropuout_rate)
        self.fc = nn.Conv2d(256, classes, kernel_size=1)

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.depthwise_conv_block(x)
        x = self.avg_pool(x)
        x = self.drop(x)
        x = self.fc(x)
        x = torch.flatten(x, 1)
        return x
