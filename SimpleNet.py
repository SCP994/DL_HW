from torch import nn
from collections import OrderedDict


class SimpleNet(nn.Module):  # 简单神经网络模型
    def __init__(self):
        super().__init__()
        self.seq1 = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 8, 6)),
            ('active1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(8, 16, 5)),
            ('active2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(16, 32, 3))
        ]))
        self.seq2 = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(80000, 400)),
            ('fc2', nn.Linear(400, 120)),
            ('fc3', nn.Linear(120, 2))
        ]))

    def forward(self, x):
        x = self.seq1(x)
        x = x.view(x.size(0), -1)
        x = self.seq2(x)
        return x
