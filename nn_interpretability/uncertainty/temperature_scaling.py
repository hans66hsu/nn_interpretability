import torch
from torch import nn, optim


class TemperatureScaling(nn.Module):
    """
    TemperatureScaling is implemented as explained in the paper
    "On Calibration of Modern Neural Networks"
    proceedings.mlr.press/v70/guo17a/guo17a.pdf

    Here we use the validation set to optimize for the T value.

    After the creation of the TS object, the object can be used
    as any other model.

    NB: The given model should return logits and not confidence estimates!
    """

    def __init__(self, model, dataset_loader, lr=0.01, iterations=100):
        """
        Creates a new TemperatureScaling object.

        :param model: The model for which TS is executed
        :param dataset_loader: The validation set for the model
        :param lr: The learning rate for the optimization process
        :param iterations: The # of optimization steps
        """
        super(TemperatureScaling, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lr = lr
        self.iterations = iterations
        self.model = model.to(self.device)
        self.T = torch.ones(1, requires_grad=True)

        print("Initial Temperature")
        print(self.T.item())

        self.set_temperature(dataset_loader)

        print("Final Temperature")
        print(self.T.item())

    def forward(self, input):
        logits = self.model(input).to(self.device)
        return logits / self.T.to(self.device).unsqueeze(1).expand(logits.size(0), logits.size(1))

    def set_temperature(self, dataset_loader):
        logits_list = []
        labels_list = []
        for input, label in dataset_loader:
            input = input.to(self.device)
            logits = self.model(input).to(self.device)
            logits_list.append(logits)
            labels_list.append(label)

        logits = nn.Parameter(torch.cat(logits_list)).to(self.device)
        labels = torch.cat(labels_list).to(self.device)

        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = optim.Adam([self.T], lr=self.lr)

        for i in range(self.iterations):
            optimizer.zero_grad()
            size = logits.shape

            output = logits / self.T.to(self.device).unsqueeze(1).expand(size[0], size[1])
            loss = criterion(output, labels).to(self.device)

            loss.backward()
            optimizer.step()

