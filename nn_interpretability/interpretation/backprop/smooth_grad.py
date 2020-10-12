import torch
from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.backprop.backprop_base import BackPropBase


class SmoothGrad(BackPropBase):
    """
    Implements the decision-based interpretability method "Guided Backpropagation"
    as outlined in the paper "Striving for Simplicity: The All Convolutional Net"
    by Springenberg et al.

    https://arxiv.org/pdf/1706.03825.pdf
    """       
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, 
                noise_level: [float], number_samples: [int]):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify.
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param noise_level: Defines the noise level of the additional Gaussian noise.
        :param number_samples: Number of samples for SmoothGrad.    
        """           
        super(SmoothGrad, self).__init__(model, classes, preprocess)
        self.noise_level = noise_level
        self.number_samples = number_samples

    def interpret(self, x):
        self.gradient = torch.zeros(x.size())
        for i in range(self.number_samples):
            noise_x = add_noise(x, self.noise_level)
            tmp_gradient = self.generate_gradient(self.model, noise_x)
            self.gradient += tmp_gradient / self.number_samples
       
        return self.gradient


def add_noise(x, noise_level: [float]):
    """
    Define add_noise out of SmoothGrad class in order to be utilized in other
    interpretability methods.
    """       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if x.size()[1] == 1:
        mean = 0;       std = noise_level * x.std().item()
        noise = torch.zeros(x.size()).normal_(mean, std)
    elif x.size(1) == 3:
        mean = 0;  noise = torch.zeros(x.size())
        for i in range(0):
            std = noise_level * x[0,i].std().item()
            noise[0,i] = noise[0, i].normal_(mean, std)
    noise_x = x + noise.to(device)
    return (noise_x.detach().to(device)).requires_grad_(True)
