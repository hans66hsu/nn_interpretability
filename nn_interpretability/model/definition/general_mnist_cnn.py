import torch.nn as nn
from collections import OrderedDict


class GeneralCNN(nn.Module):
    def __init__(self):
        super(GeneralCNN, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                           ('conv1', nn.Conv2d(1, 10, kernel_size=5)),
                           ('pool1', nn.MaxPool2d(2)),
                           ('relu1', nn.ReLU()),
                           ('conv2', nn.Conv2d(10, 20, kernel_size=5)),
                           ('pool2', nn.MaxPool2d(2)),
                           ('relu2', nn.ReLU())
                           ]))
        
        self.classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(320, 50)),
                           ('relu3', nn.ReLU()),
                           ('fc2', nn.Linear(50, 10))
                           ]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        
        return x