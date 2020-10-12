from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.backprop.backprop_base import BackPropBase


class VanillaBackprop(BackPropBase):
    """
    Implements the decision-based interpretability method "Vanilla Backpropagation"
    or "Saliency Map" as outlined in the paper "Deep Inside Convolutional Networks: 
    Visualising Image Classification Models and Saliency Maps"
    by Simonyan et al.

    https://arxiv.org/pdf/1312.6034.pdf
    """          
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify.
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """          
        super(VanillaBackprop, self).__init__(model, classes, preprocess)
    
    def interpret(self, x):
        self.gradient = self.generate_gradient(self.model, x)
        
        return self.gradient