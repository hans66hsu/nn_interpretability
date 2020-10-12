from torch.nn import Module
from torch.nn import functional as F
from torchvision import transforms
import torch

from nn_interpretability.interpretation.cam.cam_base import CAMBase


class GradCAMInterpreter(CAMBase):
    """
    Implements the decision-based interpretability method "Grad-CAM"
    as outlined in the paper "Grad-CAM: Visual Explanations from Deep Networks via
    Gradient-based Localization" by Selvaraju et al.

    https://arxiv.org/abs/1610.02391
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, input_size,
                 conv_layer_name, upsampling_mode="bilinear"):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param input_size: The size of the expected model input
        :param conv_layer_name: The name of the last conv layer
        :param upsampling_mode: The mode for the upsampling (e.g. linear, bicubic, bilinear)
        """
        CAMBase.__init__(self, model, classes, preprocess, input_size, conv_layer_name, upsampling_mode)

        self.conv_layer.register_backward_hook(self._backward_hook)

    def _backward_hook(self, module, input, output):
        self.gradient = output[0].detach().to(self.device)

    def interpret(self, x):
        x = self._execute_preprocess(x)

        model_output = self.model(x).to(self.device)
        self._last_prediction = int(torch.argmax(model_output.squeeze()).data)

        self.model.zero_grad()
        one_hot = self._one_hot(model_output, self._last_prediction)
        model_output.backward(gradient=one_hot.to(self.device), retain_graph=True)

        generated_cam = self._execute_grad_cam().to(self.device)

        return generated_cam.detach().cpu()

    def _execute_grad_cam(self):
        weights = F.adaptive_avg_pool2d(self.gradient, (1, 1)).to(self.device)

        gcam = torch.mul(self.activation.to(self.device), weights).sum(dim=1, keepdim=True).to(self.device)
        gcam = F.relu(gcam).to(self.device)
        gcam = self._upsample(gcam).to(self.device)
        gcam = (gcam - gcam.min()) / (gcam.max() - gcam.min())

        return gcam
