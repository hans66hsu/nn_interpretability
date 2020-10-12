import torch
from torch import nn
from torch.nn import Module
from torchvision.transforms import transforms
import copy

from nn_interpretability.interpretation.backprop.backprop_base import BackPropBase
from nn_interpretability.interpretation.backprop.smooth_grad import add_noise

class GuidedBackprop(BackPropBase):
    """
    Implements the decision-based interpretability method "Guided Backpropagation"
    as outlined in the paper "Striving for Simplicity: The All Convolutional Net"
    by Springenberg et al.

    https://arxiv.org/pdf/1412.6806.pdf
    """    
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, 
                 smooth=False, noise_level=0.2, number_samples=100):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify.
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param smooth: Switch for combining with SmoothGrad.
        :param noise_level: Defines the noise level of the additional Gaussian noise.
        :param number_samples: Number of samples for SmoothGrad.
        """           
        super(GuidedBackprop, self).__init__(model, classes, preprocess)
        self.model_guidedback = copy.deepcopy(model)
        self.smooth = smooth
        self.noise_level = noise_level
        self.number_samples = number_samples

    def interpret(self, x):
        relu_map = []
        # Guided Backpropagation Rule
        for name, sequential_modules in self.model_guidedback.named_children():
            if not isinstance(sequential_modules, nn.AdaptiveAvgPool2d):
                for module in sequential_modules:
                    if isinstance(module, nn.ReLU):
                        forward_hook, relu_map = self.positve_activations(relu_map)
                        module.register_forward_hook(forward_hook)
                        module.register_backward_hook(self.positive_gradients(relu_map))
        
        if self.smooth == False:
            # Guided Backpropagation
            self.gradient = self.generate_gradient(self.model_guidedback, x)
        else:
            # Guided Backpropagation + SmoothGrad
            self.gradient = torch.zeros(x.size())
            for i in range(self.number_samples):
                noise_img = add_noise(x, self.noise_level)
                tmp_gradient = self.generate_gradient(self.model_guidedback, noise_img)
                self.gradient += tmp_gradient / self.number_samples

        return self.gradient


    def positve_activations(self, activation_map):   
        def gudiedbackprop_forward(module, input, output):
            activation = output.detach().clone()
            activation[activation > 0] = 1
            activation_map.append(activation)        
        
        return gudiedbackprop_forward, activation_map


    def positive_gradients(self, a_map):  
        def guidedbackprop_backward(module, grad_in, grad_out):       
            activation = a_map.pop()
            positive_grad = grad_out[0].clamp(min=0.)
            return (activation * positive_grad, )    
    
        return guidedbackprop_backward