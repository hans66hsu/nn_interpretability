import torch
import torch.nn as nn
import random
from torch.nn import Module
from torchvision.transforms import transforms
from tqdm import tqdm

from nn_interpretability.interpretation.base.interpretation_base import ModelInterpreter
from nn_interpretability.interpretation.backprop.smooth_grad import add_noise


class DeepDream(ModelInterpreter):
    """
    Implementation of model-based interpretability method "DeepDream"
    as outlined in the paper "Inceptionism: Going deeper into neural networks"
    by Mordvintsev et al. 

    https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html
    
    Advanced techniques for better feature visualization are based on the 
    blog "Feature Visualization"
    
    https://distill.pub/2017/feature-visualization/
    """    
    def __init__(self, model: Module, preprocess: transforms.Compose,
                selected_layer: [int], selected_channel, lr, reg_term, iteration, noise, tile_size):
        ModelInterpreter.__init__(self, model, [], preprocess)
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param selected_layer: Specifies the layer as our optimization objective.
        :param selected_channel: Specifies the channel as our optimization objective. 
                                "slice(None)" means all channels
        :param lr: The learning rate of the optimization process.
        :param reg_term: The regularization term of the optimization process.
        :param iteration: The iteration for the optimization process.
        :param noise: Defines the noise level of the additional Gaussian noise.
        :param tile_size: Tile size which the image will split into, depending on GPU memory.
        """        
        self.selected_layer = selected_layer
        self.selected_channel = selected_channel
        self.lr = lr
        self.iteration = iteration
        self.reg_term = reg_term
        self.seleted_layer_grad = None
        self.tile = tile_size
        self.register_hooks()
        self.noise = noise
        
    def register_hooks(self):
        def selected_layer_hook_fn(module, grad_in, grad_out):
            self.seleted_layer_grad = grad_out[0, self.selected_channel]
        idx = self.selected_layer
        for name, sequential_modules in self.model.named_children():
            if (idx==0):
                sequential_modules.register_forward_hook(selected_layer_hook_fn)
            idx -= 1
                   
            
    def interpret(self, input_img):
        """
        Perform DeepDream by scaling input image to 8 octaves
        """            
        dream_effect = torch.zeros_like(input_img)
        for i in range(-5,3):
            tqdm.write('Processing octave {}'.format(str(i)))
            upsampled_img = nn.Upsample(scale_factor = 1.5**(i), mode='bilinear')(input_img)
            upsampled_img = add_noise(upsampled_img, self.noise).detach().cpu()
            upsampled_effect = nn.Upsample(scale_factor = 1.5**(i), mode='bilinear')(dream_effect)
            if (upsampled_img.size()[2] == 0) or (upsampled_img.size()[3] == 0):
                break
            upsampled_img = upsampled_img + upsampled_effect
            dream_img = self.dream(upsampled_img)
            dream_img = nn.Upsample(size=(input_img.size()[2],input_img.size()[3]), mode='bilinear')(dream_img)
            dream_effect = dream_img - input_img
        return dream_img
    
    def dream(self, input_img):
        """
        This is the backbone of DeepDream
        """          
        height = input_img.size()[2]
        width = input_img.size()[3]
        for i in tqdm(range(self.iteration)):
            # random roll the image (transformation robustness) to suppress noise
            shift_height, shift_width, img_rolled = self.random_roll(input_img, self.tile)
            gradients = torch.zeros_like(input_img)
            # Split the image into tiles to make dream scalable
            h_tile = range(0, height, self.tile)
            w_tile = range(0, width, self.tile)
            
            for h in h_tile:
                for w in w_tile:
                    h_end = h + self.tile if h + self.tile <= height else height
                    w_end = w + self.tile if w + self.tile <= width else width
                    img_tile = img_rolled[:, :, h:h_end, w:w_end]
                    
                    img_tile = (img_tile.detach().to(self.device)).requires_grad_(True)
                    self.model.zero_grad()
                    score = self.model(img_tile)
                    loss = self.seleted_layer_grad.mean() + self.reg_term * torch.norm(img_tile)
                    loss.backward()
                    
                    gradients_tile = img_tile.grad.detach()
                    gradients[:,:, h:h_end, w:w_end] = gradients_tile
            
            gradients = torch.roll(gradients, shifts=(-shift_height, -shift_width), dims=(2, 3))
            gradients /= gradients.std() + 1e-8
            with torch.no_grad():
                input_img += gradients * self.lr
                input_img = self.clip(input_img)
                
        return input_img.detach().cpu()
    
    def random_roll(self, input_img, max_roll, show_img = False):
        shift_height = random.randint(0, min(max_roll, input_img.size()[2]))
        shift_width = random.randint(0,  min(max_roll, input_img.size()[3]))
        img_rolled = torch.roll(input_img, shifts=(shift_height, shift_width), dims=(2, 3))
        if show_img:
            rolled = postprocess(img_rolled)
            plt.imshow(rolled)
        return shift_height, shift_width, img_rolled

    def clip(self, img):
        mean = torch.tensor([0.485, 0.456, 0.496], device=self.device)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)
        for c in range(3):
            m, s = mean[c], std[c]
            img[0, c] = torch.clamp(img[0, c], -m / s, (1 - m) / s)
        del mean, std
        return img
    