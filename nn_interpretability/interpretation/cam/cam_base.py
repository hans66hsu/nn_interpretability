from torch.nn import Module
from torch.nn import functional as F
from torchvision import transforms

from nn_interpretability.interpretation.base.interpretation_base import DecisionInterpreter


class CAMBase(DecisionInterpreter):
    """
    Base class for CAM-related interpretability methods.
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
        DecisionInterpreter.__init__(self, model, classes, preprocess)
        self.upsampling_mode = upsampling_mode
        self.input_size = input_size
        self.conv_layer_name = conv_layer_name

        self.activation = None
        self.conv_layer: Module = self.model._modules.get(self.conv_layer_name)

        if not self.conv_layer:
            raise ValueError(f"{self.conv_layer_name} is not the name of a valid layer!")

        self.conv_layer.register_forward_hook(self._forward_hook)
        self.preprocess = preprocess

    def _forward_hook(self, module, input, output):
        self.activation = output.detach().to(self.device)

    def _upsample(self, x):
        return F.interpolate(x.to(self.device), self.input_size, mode=self.upsampling_mode, align_corners=True).to(self.device)
