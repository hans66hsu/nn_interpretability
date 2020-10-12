from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.base.interpretation_base import DecisionInterpreter


class DeepLIFTRules(Enum):
    """
    Defines the different variations of the DeepLIFT method

    NoRule: Simple vanilla backpropagation
    Rescale: Rescale Rule for ReLU units
    RevealCancenl: RevealCancel Rule for ReLU units
    Linear: Linear Rule for Linear units
    LinearRescale: Both Linear and Rescale Rules
    LinearRevealCancel: Both Linear and RevealCancel Rules
    """

    NoRule = 0
    Rescale = 1
    RevealCancel = 2
    Linear = 3
    LinearRescale = 4
    LinearRevealCancel = 5


class DeepLIFT(DecisionInterpreter):
    """
    DeepLIFT is a decision-based interpretability method which tries to assign importance
    scores to every part of the input. It has been outlined in the paper
    "Learning Important Features Through Propagating Activation Differences" by Shrikumar et al.

    https://arxiv.org/pdf/1704.02685.pdf
    Appendix: http://proceedings.mlr.press/v70/shrikumar17a/shrikumar17a-supp.pdf
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, rule: DeepLIFTRules):
        """

        :param model: The model which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param rule: The DeepLIFT rule(s) which needs to be included.
        """

        DecisionInterpreter.__init__(self, model, classes, preprocess)

        self.rule = rule
        self.hooks = []

    def _setup(self):
        self._prepare_layers()

        self._replace_forward_method()
        self._register_backward_hook()

    def _register_backward_hook(self):
        device = self.device

        def register_hook(layer):
            def rescale_hook(self, input, grad_out):
                # same threshold is used in original implementation
                # https://github.com/kundajelab/deeplift/blob/7cb4804cc8e682662652ae24903d9492b4d74523/deeplift/util.py#L14
                # Data points with values lower than the threshold do not use the multiplier, but the gradient directly.
                near_zero_threshold = 1e-7
                alpha = 0.01

                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)

                ref_y = self.outputs[0].to(device)
                y = self.outputs[1].to(device)
                delta_y = (y - ref_y).to(device)

                multiplier = (delta_y / (delta_x + 1e-7)).to(device)

                far_zero_contrib_mask = (delta_x.abs() > near_zero_threshold).float().to(device)
                with_multiplier = far_zero_contrib_mask * multiplier
                with_multiplier = with_multiplier.to(device)

                near_zero_contrib_mask = (delta_x.abs() <= near_zero_threshold).float().to(device)
                without_multiplier = near_zero_contrib_mask * alpha
                without_multiplier = without_multiplier.to(device)

                scale_factor = with_multiplier + without_multiplier
                output = (scale_factor * input[0]).to(device)

                return (output,)

            def reveal_cancel_hook(self, input, grad_out):
                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)
                delta_x_plus = ((delta_x >= 0).float().to(device))*delta_x
                delta_x_minus = ((delta_x < 0).float().to(device))*delta_x

                delta_y_plus = 0.5*(F.relu(ref_x + delta_x_plus) - F.relu(ref_x)) +\
                               0.5*(F.relu(ref_x + delta_x_plus + delta_x_minus) - F.relu(ref_x + delta_x_minus))

                delta_y_minus = 0.5 * (F.relu(ref_x + delta_x_minus) - F.relu(ref_x)) + \
                                0.5 * (F.relu(ref_x + delta_x_plus + delta_x_minus) - F.relu(ref_x + delta_x_plus))

                m_x_plus = delta_y_plus / (delta_x_plus + 1e-7)
                m_x_plus *= 1
                m_x_minus = delta_x_minus / (delta_y_minus + 1e-7)
                m_x_minus *= 1

                grad = input[0]
                grad_plus = ((grad >= 0).float().to(device)) * grad
                grad_minus = ((grad < 0).float().to(device)) * grad

                output = grad_plus * m_x_plus + grad_minus * m_x_minus

                return (output,)

            def linear_hook(self, input, grad_out):
                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)
                delta_x_plus = ((delta_x > 0).float().to(device))
                delta_x_minus = ((delta_x < 0).float().to(device))
                delta_x_zero = ((delta_x == 0).float().to(device))

                transposed_weight = self.weight.detach().clone().T.to(device)
                size = transposed_weight.size()
                transpose_pos = nn.Linear(size[1], size[0]).to(device)
                transpose_pos.weight = nn.Parameter(((transposed_weight > 0).float().to(device)) * transposed_weight)
                transpose_negative = nn.Linear(size[1], size[0]).to(device)
                transpose_negative.weight = nn.Parameter(((transposed_weight < 0).float().to(device)) * transposed_weight)

                transpose_full = nn.Linear(size[1], size[0]).to(device)
                transpose_full.weight = nn.Parameter(transposed_weight)

                ref_y = self.outputs[0].to(device)
                y = self.outputs[1].to(device)
                delta_y = (y - ref_y).to(device)
                delta_y_plus = ((delta_y > 0).float().to(device)) * delta_y
                delta_y_minus = ((delta_y < 0).float().to(device)) * delta_y

                pos_grad_out = delta_y_plus * grad_out[0]
                neg_grad_out = delta_y_minus * grad_out[0]

                pos_pos_result = transpose_pos.forward(pos_grad_out) * delta_x_plus
                pos_neg_result = transpose_pos.forward(neg_grad_out) * delta_x_plus
                neg_pos_result = transpose_negative.forward(neg_grad_out) * delta_x_minus
                neg_neg_result = transpose_negative.forward(pos_grad_out) * delta_x_minus
                null_result = transpose_full.forward(grad_out[0]) * delta_x_zero

                multiplier = pos_pos_result + pos_neg_result + neg_pos_result + neg_neg_result + null_result

                out = (input[0],) + (multiplier.to(device),) + input[2:]
                return out

            def linear_conv_hook(self, input, grad_out):
                ref_x = self.inputs[0].to(device)
                x = self.inputs[1].to(device)
                delta_x = (x - ref_x).to(device)
                delta_x_plus = ((delta_x > 0).float().to(device))
                delta_x_minus = ((delta_x < 0).float().to(device))
                delta_x_zero = ((delta_x == 0).float().to(device))

                transpose_pos = nn.ConvTranspose2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.padding).to(device)
                transpose_pos.weight = nn.Parameter(((self.weight > 0).float().to(device))*self.weight.detach().clone().to(device)).to(device)

                transpose_negative = nn.ConvTranspose2d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.padding).to(device)
                transpose_negative.weight = nn.Parameter(((self.weight < 0).float().to(device)) * self.weight.detach().clone()).to(device)

                transpose_full = nn.ConvTranspose2d(self.out_channels, self.in_channels, self.kernel_size,
                                                        self.stride, self.padding).to(device)
                transpose_full.weight = nn.Parameter((self.weight.detach().clone().to(device))).to(device)

                ref_y = self.outputs[0].to(device)
                y = self.outputs[1].to(device)
                delta_y = (y - ref_y).to(device)
                delta_y_plus = ((delta_y > 0).float().to(device)) * delta_y
                delta_y_minus = ((delta_y < 0).float().to(device)) * delta_y

                pos_grad_out = delta_y_plus * grad_out[0]
                neg_grad_out = delta_y_minus * grad_out[0]

                dim_check = transpose_pos.forward(pos_grad_out)
                if dim_check.shape != delta_x.shape:
                    if dim_check.shape[3] > delta_x.shape[3]:
                        dim_diff = dim_check.shape[3] - delta_x.shape[3]
                        delta_x = torch.cat((delta_x, torch.ones(delta_x.shape[0], delta_x.shape[1], dim_diff, delta_x.shape[3])), 2)
                        delta_x = torch.cat((delta_x, torch.ones(delta_x.shape[0], delta_x.shape[1], delta_x.shape[2], dim_diff)), 3)
                    else:
                        new_shape = dim_check.shape
                        delta_x = delta_x[0:new_shape[0], 0:new_shape[1], 0:new_shape[2], 0:new_shape[3]]

                    delta_x_plus = ((delta_x > 0).float().to(device))
                    delta_x_minus = ((delta_x < 0).float().to(device))
                    delta_x_zero = ((delta_x == 0).float().to(device))

                pos_pos_result = transpose_pos.forward(pos_grad_out) * delta_x_plus
                pos_neg_result = transpose_pos.forward(neg_grad_out) * delta_x_plus
                neg_pos_result = transpose_negative.forward(neg_grad_out) * delta_x_minus
                neg_neg_result = transpose_negative.forward(pos_grad_out) * delta_x_minus
                null_result = transpose_full.forward(grad_out[0]) * delta_x_zero

                multiplier = pos_pos_result + pos_neg_result + neg_pos_result + neg_neg_result + null_result

                if input[0].shape != multiplier.shape:
                    if input[0].shape[3] > multiplier.shape[3]:
                        dim_diff = input[0].shape[3] - multiplier.shape[3]
                        multiplier = torch.cat((multiplier, torch.ones(multiplier.shape[0], multiplier.shape[1], dim_diff, multiplier.shape[3])), 2)
                        multiplier = torch.cat((multiplier, torch.ones(multiplier.shape[0], multiplier.shape[1], multiplier.shape[2], dim_diff)), 3)
                    else:
                        new_shape = input[0].shape
                        multiplier = delta_x[0:new_shape[0], 0:new_shape[1], 0:new_shape[2], 0:new_shape[3]]

                out = (multiplier.to(device),) + input[1:]
                return out

            if self.rule != DeepLIFTRules.NoRule:
                if isinstance(layer, torch.nn.ReLU) and self.rule != DeepLIFTRules.Linear:
                    if self.rule == DeepLIFTRules.LinearRescale or self.rule == DeepLIFTRules.Rescale:
                        self.hooks.append(layer.register_backward_hook(rescale_hook))
                    else:
                        self.hooks.append(layer.register_backward_hook(reveal_cancel_hook))
                elif isinstance(layer, torch.nn.Linear) and \
                        (self.rule == DeepLIFTRules.LinearRescale or self.rule == DeepLIFTRules.LinearRevealCancel or self.rule == DeepLIFTRules.Linear):
                    self.hooks.append(layer.register_backward_hook(linear_hook))
                elif isinstance(layer, torch.nn.Conv2d) and \
                        (self.rule == DeepLIFTRules.LinearRescale or self.rule == DeepLIFTRules.LinearRevealCancel or self.rule == DeepLIFTRules.Linear):
                    self.hooks.append(layer.register_backward_hook(linear_conv_hook))

        self.model.apply(register_hook)

    def _replace_forward_method(self):
        device = self.device

        def add_forward_hook(layer):
            def forward_hook(self, input, output):
                self.inputs.append(input[0].data.clone().to(device))
                self.outputs.append(output[0].data.clone().to(device))

            if self.rule != DeepLIFTRules.NoRule:
                self.hooks.append(layer.register_forward_hook(forward_hook))

        self.model.apply(add_forward_hook)

    def _prepare_layers(self):
        def _init_layers(layer):
            layer.inputs = []
            layer.outputs = []

        self.model.apply(_init_layers)

    def _generate_baseline(self, x):
        self.model(torch.zeros(x.shape).to(self.device)).to(self.device)

    def interpret(self, x):
        try:
            x = self._execute_preprocess(x)

            self._setup()

            self._generate_baseline(x)

            x = x.to(self.device)
            x.requires_grad = True

            output = self.model(x).to(self.device)
            self.model.zero_grad()

            self._last_prediction = output.argmax().item()
            grad_out = torch.zeros(output.shape).to(self.device)
            grad_out[0][self._last_prediction] = 1.0
            output.backward(grad_out)

            self.cleanup()

            return x.grad.data.to(self.device).detach().cpu()
        finally:
            self.cleanup()

    def cleanup(self):
        for hook in self.hooks:
            hook.remove()

        self.hooks = []
