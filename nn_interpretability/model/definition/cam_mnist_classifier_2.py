import torch.nn as nn
from torch.nn.functional import pad


class CAMMNISTExtendedClassifier(nn.Module):
    def __init__(self):
        super(CAMMNISTExtendedClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.relu3 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(pad(self.relu1(self.conv1(x)), [2, 2, 2, 2]))
        x = self.pool2(pad(self.relu2(self.conv2(x)), [2, 2, 2, 2]))
        x = self.relu3(self.conv3(x))

        x = self.avgpool(x)

        x = x.view(x.shape[0], -1)
        return self.fc(x)
