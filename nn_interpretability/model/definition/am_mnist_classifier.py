import torch.nn as nn
import torch.nn.functional as F


class AMCNN(nn.Module):
    def __init__(self):
        super(AMCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
