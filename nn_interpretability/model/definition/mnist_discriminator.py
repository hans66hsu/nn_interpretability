import torch.nn as nn


class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()
        self.dense1 = nn.Linear(28*28, 128, True)
        self.dense2 = nn.Linear(128, 1, True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = nn.functional.relu(self.dense1(x))
        x = nn.functional.sigmoid(self.dense2(x))

        return x
