from torch.nn import Module
from torchvision import transforms
import torch
import numpy as np

from nn_interpretability.interpretation.cam.cam_base import CAMBase


class CAMInterpreter(CAMBase):
    """
    Implements the decision-based interpretability method "Class Activation Maps (CAM)"
    as outlined in the paper "Learning Deep Features for Discriminative Localization"
    by Zhou et al.

    http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf
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

        self.classification_weights = np.squeeze(list(self.model.parameters())[-2].data.cpu().numpy())

    def interpret(self, x):
        x = self._execute_preprocess(x)
        model_output = self.model(x).to(self.device)
        model_output = model_output.squeeze()

        self._last_prediction = int(torch.argmax(model_output).data)
        generated_cam = self._execute_cam(self._last_prediction)

        return generated_cam.detach().cpu()

    def _execute_cam(self, class_index: int):
        bz, nc, h, w = self.activation.cpu().numpy().shape

        cam = np.einsum("i,ikm->km", self.classification_weights[class_index], self.activation.cpu().reshape((nc, h, w)))
        cam = cam.reshape(1, 1, h, w)
        cam = cam - cam.min()
        cam = cam / cam.max()

        cam_img = self._upsample(torch.Tensor(cam).to(self.device)).to(self.device)

        return cam_img
