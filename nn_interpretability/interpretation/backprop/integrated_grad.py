import torch
from torch import nn
from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.backprop.backprop_base import BackPropBase
from nn_interpretability.interpretation.backprop.smooth_grad import add_noise

class IntegratedGrad(BackPropBase):
    """
    Implements the decision-based interpretability method "Integrated Gradients"
    as outlined in the paper "Axiomatic Attribution for Deep Networks"
    by Sundararajan et al.

    https://arxiv.org/pdf/1703.01365.pdf
    """      
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, baseline,
                steps: [int], smooth=False, noise_level=0.2):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify.
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param baseline: Define the baseline. Normally a black image.
        :param steps: Number of steps to approximate integral.
        :param smooth: Switch for combining with SmoothGrad.
        :param noise_level: Defines the noise level of the additional Gaussian noise.
        """         
        super(IntegratedGrad, self).__init__(model, classes, preprocess)
        self.steps = steps
        self.smooth = smooth
        self.noise_level = noise_level
        self.dist_list = []
        self.baseline = baseline.to(self.device)

    def interpret(self, x):
        # Inegrated Gradients Rule
        self.linear_path(x)
        self.gradient = torch.zeros(x.size())

        for dist in self.dist_list:
            tmp_gradient = self.generate_gradient(self.model, dist)
            self.gradient += tmp_gradient / self.steps
       
        return self.gradient

    def linear_path(self, x):
        step_list = []
        step_list = range(0, self.steps + 1)
        dist = x - self.baseline
        # Generate interpolated images between baseline and input image
        for step in step_list:
            if self.smooth == False:
                # Integraged Gradients
                inter_img = (self.baseline + (dist * step / self.steps)).detach().requires_grad_(True)
            else:
                # Integrated Gradients + SmoothGrad
                inter_img = add_noise((self.baseline + (dist * step / self.steps)), self.noise_level)
            self.dist_list.append(inter_img)

  