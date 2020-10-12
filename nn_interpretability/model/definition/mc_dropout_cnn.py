import torch.nn as nn
from collections import OrderedDict


class CNN_Dropout(nn.Module):
    def __init__(self, drop_rate=0.5):
        super(CNN_Dropout, self).__init__()
        self.features = nn.Sequential(OrderedDict([
                           ('conv1', nn.Conv2d(1,10, kernel_size=5)),
                           ('dropout1', nn.Dropout2d(p=drop_rate)),
                           ('pool1', nn.MaxPool2d(2)),
                           ('relu1', nn.ReLU()),
                           ('conv2', nn.Conv2d(10,20, kernel_size=5)),
                           ('dropout2', nn.Dropout2d(p=drop_rate)),
                           ('pool2', nn.MaxPool2d(2)),
                           ('relu2', nn.ReLU())
                           ]))
        
        
        self.classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(320, 50)),
                           ('relu3', nn.ReLU()),
                           ('dropout3', nn.Dropout(p=drop_rate)),
                           ('fc2', nn.Linear(50, 10))
                           ]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        
        return x