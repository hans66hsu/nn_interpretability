import torch

import torch.nn as nn
from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.deconv.deconv_base import DeconvolutionBase
from nn_interpretability.interpretation.lrp.lrp_base import LRPBase


class LRPTranspose(LRPBase, DeconvolutionBase):
    """
    Implements LRP-0 by using Transposed layers.
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, visualize_layer=0):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param visualize_layer: The number of the layer for which the relevance scores should be visualized.
        """
        super(LRPTranspose, self).__init__(model, classes, preprocess)
        super(DeconvolutionBase, self).__init__(model, classes, preprocess)

        self.visualize_layer = visualize_layer
        self.LRP_layers = self.layers

        # Relevance scores
        self.Relevance = []

        # Activations
        self.A = []

    def _execute_model_forward_pass(self, x):
        max_pool_indices = []
        should_rescale = True
        prev_size = -1
        view_resize = -1

        for layer in self.layers:
            if isinstance(layer, nn.Linear) and should_rescale:
                last_activation = self.A.pop()
                prev_size = last_activation.size()
                view_resize = layer.weight.size()[1]
                new_activation = last_activation.view(-1, layer.weight.size()[1]).to(self.device)
                self.A.append(new_activation)
                should_rescale = False

            if isinstance(layer, nn.MaxPool2d):
                layer.return_indices = True
                activation, indices = layer.forward(self.A[-1])
                max_pool_indices.append(indices.to(self.device))
                layer.return_indices = False
            else:
                activation = layer.forward(self.A[-1])

            activation = activation.to(self.device)
            self.A.append(activation)

        return max_pool_indices, prev_size, view_resize

    def _execute_lrp_backward_pass(self, max_pool_indices, prev_size, view_resize):
        self.A.reverse()

        for key, transposed_layer in enumerate(self.transposed_layers):
            layer = self.layers[len(self.layers) - key - 1]

            z = layer.forward(self.A[key + 1]) + 1e-7
            s = (self.Relevance[-1] / z).detach()

            if isinstance(transposed_layer, nn.MaxUnpool2d):
                idx = max_pool_indices.pop()
                c = transposed_layer.forward(s, idx)
            else:
                c = transposed_layer.forward(s)

            relevance = (self.A[key + 1] * c).detach()
            if relevance.size()[1] == view_resize:
                relevance = relevance.view(-1, prev_size[1], prev_size[2], prev_size[3])

            self.Relevance.append(relevance)

        self.A.reverse()
        self.Relevance.reverse()

    def interpret(self, x):
        x = self._execute_preprocess(x)

        self.A = [x]
        self.Relevance = []

        max_pool_indices, prev_size, view_resize = self._execute_model_forward_pass(x)

        self._last_prediction = torch.argmax(self.A[-1]).to(self.device).item()
        first_relevance = torch.zeros(1, 10).to(self.device)
        first_relevance[0][self._last_prediction] = self.A[-1].data[0][self._last_prediction]
        self.Relevance.append(first_relevance)

        self._execute_lrp_backward_pass(max_pool_indices, prev_size, view_resize)

        if self.visualize_layer == 0:
            self.lrp_pixel()

        return self.Relevance[self.visualize_layer].detach().cpu()
