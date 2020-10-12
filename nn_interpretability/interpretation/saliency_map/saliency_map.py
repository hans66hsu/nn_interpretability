import torch
from torch.nn import Module
from torchvision import transforms

from nn_interpretability.interpretation.base.interpretation_base import DecisionInterpreter


class SaliencyMap(DecisionInterpreter):
    """
    Implements the decision-based interpretability method "Vanilla Backpropagation"
    or "Saliency Map" as outlined in the paper "Deep Inside Convolutional Networks: 
    Visualising Image Classification Models and Saliency Maps"
    by Simonyan et al.

    https://arxiv.org/pdf/1312.6034.pdf
    """         
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, image_overlay=False):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify.
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param image_overlay: Switch for image x gradients
        """         
        DecisionInterpreter.__init__(self, model, classes, preprocess)
        self.image_overlay = image_overlay

    def interpret(self, x):
        x.requires_grad = True

        x.retain_grad()
        scores = self.model(x)

        score_max_index = scores.argmax()
        score_max = scores[0, score_max_index]
        score_max.backward()

        saliency_map = torch.max(x.grad.detach(), dim=1)[0]

        if self.image_overlay:
            saliency_map = saliency_map * x

        saliency_map = self.normalize(saliency_map)
        return saliency_map
