import torch
import torch.nn as nn
from torch.nn import Module
from torchvision import transforms

from nn_interpretability.interpretation.deconv.deconv_base import DeconvolutionBase


class DeconvolutionPartialReconstruction(DeconvolutionBase):
    """
    Partial Input Reconstruction Deconvolution is a decision-based interpretability method
    which aims to partially recreate the input from the output of the model by using only
    a single filter in a layer of choice. The procedure is executed for every filter
    in the chosen layer.
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, layer_number):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param layer_number: The number of the convolutional layer for which the procedure should be executed.
                        For example, 1 for the first CONV layer. 2 for the second CONV layer and so on.
        """
        DeconvolutionBase.__init__(self, model, classes, preprocess)
        self.layer_number = layer_number

        if self.layer_number <= 0:
            raise ValueError("Layer number can not be negative!")

    def interpret(self, x):
        x = self._execute_preprocess(x)
        results = []

        layer_index = -1
        counter = self.layer_number
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Conv2d):
                counter -= 1
                if counter == 0:
                    layer_index = i
                    break

        if layer_index < 0:
            raise ValueError("Layer number is not valid!")

        filters_count = self.layers[layer_index].weight.size()[0]
        for i in range(filters_count):
            new_weights = torch.zeros(self.layers[layer_index].weight.size()).to(self.device)
            new_weights[i] = self.layers[layer_index].weight[i].clone().to(self.device)
            self.transposed_layers[len(self.transposed_layers) - layer_index - 1].weight = torch.nn.Parameter(new_weights).to(self.device)

            y, max_pool_indices, prev_size, view_resize = self._execute_model_forward_pass(x)
            y = self._execute_transposed_model_forward_pass(y, max_pool_indices, prev_size, view_resize)
            y = y.detach().cpu()
            y = (y - y.min()) / (y.max() - y.min())
            results.append(y)

        return results

