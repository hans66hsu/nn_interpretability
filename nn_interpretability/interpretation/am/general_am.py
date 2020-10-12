import torch
from torch import nn
from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.am.am_base import ActivationMaximizationBase


class ActivationMaximization(ActivationMaximizationBase):
    """
    Implements the model-based interpretability method "General Activation maximization"
    as outlined in the paper "Methods for Interpreting and Understanding Deep Neural Networks"
    by Montavon et al.

    https://arxiv.org/abs/1706.07979
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, input_size,
                 start_img, class_num, lr, reg_term, class_mean, epochs, verbose=False, threshold=0.999):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param input_size: The expected input size by the given model
        :param start_img: The starting point for the optimization process
        :param class_num: The class for which it is optimized
        :param lr: The learning rate of the optimization process
        :param reg_term: The regularization term of the optimization process
        :param class_mean: The mean of the class for which we optimize
        :param epochs: The maximum amount of epochs for the optimization process
        :param verbose: Defines the verbosity of logging
        :param threshold: The threshold after which the optimization process should stop (e.g. 99.99%)
        """
        ActivationMaximizationBase.__init__(self, model, classes, preprocess, input_size)

        self.start_img = start_img.to(self.device)
        self.start_img = self._execute_preprocess(self.start_img)
        self.class_num = class_num
        self.lr = lr
        self.reg_term = reg_term
        self.epochs = epochs
        self.class_mean = class_mean.to(self.device)
        self.verbose = verbose
        self.threshold = threshold

    def interpret(self):
        gen_img = self.start_img.clone().detach().to(self.device)
        gen_img.requires_grad = True

        optimizer = torch.optim.Adam([gen_img], lr=self.lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss().to(self.device)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            scores = self.model(gen_img).to(self.device)
            probs = torch.nn.functional.softmax(scores, 1)[0]
            targets = torch.LongTensor([self.class_num]).to(self.device)
            loss = criterion(scores, targets).to(self.device)
            loss_reg = (torch.sqrt(((gen_img - self.class_mean)**2).sum()) * self.reg_term).to(self.device)
            loss = loss + loss_reg

            if probs[self.class_num] >= self.threshold:
                if self.verbose:
                    print(f'Class {self.class_num} | Epoch {epoch} | Loss {loss} | Probability {probs[self.class_num]}')
                break

            if self.verbose and epoch % 100 == 0:
                print(f'Class {self.class_num} | Epoch {epoch} | Loss {loss} | Probability {probs[self.class_num]}')

            loss.backward()
            optimizer.step()

        return gen_img.detach().cpu()

