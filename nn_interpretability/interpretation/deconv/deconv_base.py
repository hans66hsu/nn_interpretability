import torch.nn as nn
from torch.nn import Module
from torchvision import transforms

from nn_interpretability.interpretation.base.interpretation_base import DecisionInterpreter


def transpose(layer):
    transpose = None

    if isinstance(layer, nn.Linear):
        transposed_weight = layer.weight.detach().clone().T
        size = transposed_weight.size()
        transpose = nn.Linear(size[1], size[0])
        transpose.weight = nn.Parameter(transposed_weight)
    elif isinstance(layer, nn.ReLU):
        transpose = nn.ReLU()
    elif isinstance(layer, nn.Conv2d):
        transpose = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, layer.kernel_size, layer.stride,
                                       layer.padding)
        transpose.weight = nn.Parameter(layer.weight.detach().clone())
    elif isinstance(layer, nn.MaxPool2d):
        transpose = nn.MaxUnpool2d(layer.kernel_size)
    else:
        print(layer)
        raise ValueError("The network have a layer type which is currently not supported.")

    return transpose


class DeconvolutionBase(DecisionInterpreter):
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """
        DecisionInterpreter.__init__(self, model, classes, preprocess)
        self.transposed_layers = []
        self.layers = []

        def apply_method(layer):
            if len(layer._modules) == 0:
                self.layers.append(layer.to(self.device))
                transposed_layer = transpose(layer).to(self.device)
                self.transposed_layers.append(transposed_layer)

        self.model.apply(apply_method)

        self.transposed_layers.reverse()

    def _execute_model_forward_pass(self, x):
        max_pool_indices = []
        should_rescale = True
        prev_size = -1
        view_resize = -1

        for layer in self.layers:
            if isinstance(layer, nn.Linear) and should_rescale:
                prev_size = x.size()
                view_resize = layer.weight.size()[1]
                x = x.view(-1, layer.weight.size()[1])
                should_rescale = False

            if isinstance(layer, nn.MaxPool2d):
                layer.return_indices = True
                x, indices = layer.forward(x)
                max_pool_indices.append(indices.to(self.device))
                layer.return_indices = False
            else:
                x = layer.forward(x)

            x =  x.to(self.device)

        return x, max_pool_indices, prev_size, view_resize

    def _execute_transposed_model_forward_pass(self, y, max_pool_indices, prev_size, view_resize):
        for counter, transposed_layer in enumerate(self.transposed_layers):
            if isinstance(transposed_layer, nn.MaxUnpool2d):
                idx = max_pool_indices.pop().to(self.device)
                y = transposed_layer.forward(y, idx).to(self.device)
            else:
                y = transposed_layer.forward(y).to(self.device)

            if y.size()[1] == view_resize:
                y = y.reshape(prev_size)

        return y.to(self.device)
