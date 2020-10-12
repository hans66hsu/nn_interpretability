import torch
import numpy as np

from torch.nn import Module
from torchvision.transforms import transforms
from nn_interpretability.interpretation.lrp.lrp_base import LRPBase


class LRPAlphaBeta(LRPBase):
    """
    Implements the decision-based interpretability method "LRP-ab"
    as outlined in the paper "Methods for Interpreting and Understanding Deep Neural Networks"
    by Montavon et al.

    https://arxiv.org/pdf/1706.07979.pdf
    """      
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, 
                 alpha, beta, visualize_layer=0, input_z_beta=True, treat_avgpool=False):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify.
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param alpha: alpha number for LRP-ab rule.
        :param beta: beta number for LRP-ab rule.
        :param visualize_layer: Select the layer we want to visualize heatmap.
        :param input_z_beta: Switch for using LRP z^beta rule at input layers.
        :param treat_avgpool: Switch for treat max pooling like average pooling described in Montavon's paper.
        """         
        super(LRPAlphaBeta, self).__init__(model, classes, preprocess)
        self.alpha = alpha
        self.beta = beta
        self.visualize_layer = visualize_layer
        self.input_z_beta = input_z_beta
        self.treat_avgpool = treat_avgpool

    def interpret(self, x):
        # Create a list to collect all layers
        x = x.detach().to(self.device)
        self.A = [x]
        layers = self.to_conv(x)
        L = len(layers)
        for l in range(L):
            self.A.append(layers[l].forward(self.A[l]))
        
        _, self.predicted_class = torch.max(self.A[-1], dim=1)
        if self.classes == 'predicted':
            self.classes = self.predicted_class.item()
        
        # Relevance score
        num_cls = self.A[-1].size(1)
        T = torch.FloatTensor((1.0 * (np.arange(num_cls) == self.classes).reshape([1,num_cls,1,1])))
        self.Relevance = [None] * L + [self.A[-1].detach() * T.to(self.device)]
        
        # Uncomment this and comment "self.LRP_layers = layers" if you have enough GPU memory. 
        # This is to make sure that when we treat max pooling as average pooling we don't change the original layer
        #self.LRP_layers = copy.deepcopy(layers) 
        self.LRP_layers = layers
    
        # LRP　backward passincr_p
        for l in range(self.visualize_layer, L)[::-1]:
            self.A[l] = (self.A[l].detach().to(self.device)).requires_grad_(True)
            rho_p = lambda p: p.clamp(min=0.);         incr_p = lambda z: z+1e-9
            rho_n = lambda p: p.clamp(max=0.);         incr_n = lambda z: z-1e-9

            if isinstance(self.LRP_layers[l],torch.nn.MaxPool2d) or \
                isinstance(self.LRP_layers[l],torch.nn.AdaptiveAvgPool2d):
                if self.treat_avgpool:
                    #treat max pooling like average pooling described in Montavon's paper
                    self.LRP_layers[l] = torch.nn.AvgPool2d(2)
            
                z = incr_p(self.LRP_layers[l].forward(self.A[l]))            # step 1            
                s = (self.Relevance[l+1] / z).detach()                       # step 2            
                (z * s).sum().backward(); c = self.A[l].grad                 # step 3
                self.Relevance[l] = (self.A[l] * c).detach()                 # step 4            

            elif isinstance(self.LRP_layers[l],torch.nn.Conv2d):
                z_p = incr_p(self.newlayer(self.LRP_layers[l], rho_p).forward(self.A[l]))           # step 1
                s_p = (self.Relevance[l+1]/z_p).detach()                                            # step 2
                (z_p*s_p).sum().backward(); c_p = self.A[l].grad                                    # step 3
                
                self.A[l] = (self.A[l].detach().to(self.device)).requires_grad_(True)
                
                z_n = incr_n(self.newlayer(self.LRP_layers[l], rho_n).forward(self.A[l]))           # step 1
                s_n = (self.Relevance[l+1]/z_n).detach()                                            # step 2
                (z_n*s_n).sum().backward(); c_n = self.A[l].grad                                    # step 3

                self.Relevance[l] = (self.A[l] * ((self.alpha * c_p) - (self.beta * c_n))).detach() # step 4

            else:
                self.Relevance[l] = self.Relevance[l+1]
        
        if (self.input_z_beta == True) and (self.visualize_layer == 0): self.lrp_pixel()

        return self.Relevance[self.visualize_layer]
    




