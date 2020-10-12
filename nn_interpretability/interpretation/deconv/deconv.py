from torch.nn import Module
from torchvision import transforms

from nn_interpretability.interpretation.deconv.deconv_base import DeconvolutionBase


class Deconvolution(DeconvolutionBase):
    """
    Deconvolution is a decision-based interpretability method, which aims to reconstruct the input
    based on the output of the model.

    The implementation is based on the paper "Visualizing and Understanding Convolutional Networks"
    by Zeiler et al.

    https://arxiv.org/pdf/1311.2901.pdf
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """
        DeconvolutionBase.__init__(self, model, classes, preprocess)

    def interpret(self, x):
        x = self._execute_preprocess(x)

        y, max_pool_indices, prev_size, view_resize = self._execute_model_forward_pass(x)
        y = self._execute_transposed_model_forward_pass(y, max_pool_indices, prev_size, view_resize)
        y = y.detach().cpu()
        y = (y - y.min()) / (y.max() - y.min())

        return y

