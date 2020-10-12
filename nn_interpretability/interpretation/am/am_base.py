from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.base.interpretation_base import ModelInterpreter


class ActivationMaximizationBase(ModelInterpreter):
    """
    Base class for Activation Maximization interpretability methods.
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, input_size):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param input_size: The expected input size by the given model
        """
        ModelInterpreter.__init__(self, model, classes, preprocess)
        self.input_size = input_size
