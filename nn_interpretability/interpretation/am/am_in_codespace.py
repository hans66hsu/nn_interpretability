import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.am.am_base import ActivationMaximizationBase


class ActivationMaximizationCodespace(ActivationMaximizationBase):
    """
    Implements the model-based interpretability method "Activation maximization in CodeSpace"
    as outlined in the paper "Methods for Interpreting and Understanding Deep Neural Networks"
    by Montavon et al.

    https://arxiv.org/abs/1706.07979
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, input_size,
                 class_num, lr, reg_term, epochs, generator: Module, latent_dim, class_mean, verbose=False,
                 threshold=0.99, is_dcgan=False):
        """

        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param input_size: The expected input size by the given model
        :param class_num: The class for which it is optimized
        :param lr: The learning rate of the optimization process
        :param reg_term: The regularization term of the optimization process
        :param epochs: The maximum amount of epochs for the optimization process
        :param generator: The generative model used for generating images
        :param latent_dim: The size of the latent space in which we optimize
        :param class_mean: The mean value of the class for which we optimize
        :param verbose: Defines the verbosity of logging.
        :param threshold: The threshold after which the optimization process should stop (e.g. 99.99%)
        :param is_dcgan: Flag is true if the generator is a DCGAN network.
        """
        super(ActivationMaximizationCodespace, self).__init__(model, classes, preprocess, input_size)

        self.class_num = class_num
        self.lr = lr
        self.reg_term = reg_term
        self.epochs = epochs
        self.generator = generator
        self.latent_dim = latent_dim
        self.class_mean = class_mean.to(self.device)
        self.verbose = verbose
        self.threshold = threshold
        self.is_dcgan = is_dcgan

    def interpret(self):
        z = torch.autograd.Variable(torch.FloatTensor(np.random.normal(0, 1, (1, self.latent_dim))))
        if self.is_dcgan:
            z = torch.randn(1, self.latent_dim, 1, 1)

        z = z.to(self.device)
        z.requires_grad = True

        optimizer = torch.optim.Adam([z], lr=self.lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss().to(self.device)

        gen_img = torch.zeros(self.input_size)
        gen_img = gen_img.to(self.device)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            gen_img = self.generator(z).to(self.device)
            scores = self.model(gen_img).to(self.device)
            probs = F.softmax(scores, 1)[0]
            targets = torch.LongTensor([self.class_num]).to(self.device)

            loss_without_reg = criterion(scores, targets).to(self.device)
            loss_reg = (torch.sqrt(((gen_img - self.class_mean)**2).sum()) * self.reg_term).to(self.device)
            loss = loss_without_reg + loss_reg

            if probs[self.class_num] >= self.threshold:
                if self.verbose:
                    print(f'Class {self.class_num} | Epoch {epoch} | Loss {loss} | Probability {probs[self.class_num]}')
                break

            loss.backward()
            optimizer.step()

            if self.verbose and epoch % 100 == 0:
                print(f'Class {self.class_num} | Epoch {epoch} | Loss {loss} | Probability {probs[self.class_num]}')

        return gen_img.detach().cpu()
