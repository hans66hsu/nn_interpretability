import torch.nn as nn
import copy

from torch.nn import Module
from torchvision.transforms import transforms
from nn_interpretability.interpretation.base.interpretation_base import DecisionInterpreter


class LRPBase(DecisionInterpreter):
    """
    Base class for LRP interpretability methods.
    """     
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """
        super(LRPBase, self).__init__(model, classes, preprocess)
        self.Relevance = None
        self.A = None
        self.LRP_layers = None
        self.predicted_class = None
        
    def get_class(self):
        return self.predicted_class.item()
    
    def lrp_pixel(self):
        """
        LRP z^beta rule for input layers in pixel domain
        """        
        self.A[0] = (self.A[0].detach()).requires_grad_(True)
        
        lb = (self.A[0].detach()*0-1).requires_grad_(True)
        hb = (self.A[0].detach()*0+1).requires_grad_(True)
        
        rho_p = lambda p: p.clamp(min=0); 
        rho_n = lambda p: p.clamp(max=0); 

        positive_layers = self.newlayer(self.LRP_layers[0], rho_p)
        negative_layers = self.newlayer(self.LRP_layers[0], rho_n)
        
        z = self.LRP_layers[0].forward(self.A[0])+ 1e-9                 # step 1
        z -= positive_layers.forward(lb)
        z -= negative_layers.forward(hb)
        
        s = (self.Relevance[1]/z).detach()                              # step 2
        
        (z*s).sum().backward(); c = self.A[0].grad                      # step 3
        c = self.A[0].grad;  cp = lb.grad;  cm = hb.grad 
        
        #LRP z-beta rule for first layer pixels, suggested from Montavon's paper
        self.Relevance[0] = (self.A[0]*c + lb*cp + hb*cm).detach()      # step 4
                
    def newlayer(self, layer, g):
        output_layer = copy.deepcopy(layer)
        try: 
            if layer.weight is not None: output_layer.weight = nn.Parameter(g(layer.weight));
        except AttributeError: pass

        try: 
            if layer.bias is not None:  output_layer.bias   = nn.Parameter(g(layer.bias))
        except AttributeError: pass

        return output_layer

    def to_conv(self, x):
        """
        Transfer FC layers to Conv layers. 
        Note: This enable us to visualize the heatmap from FC layers. 
              We have compared the results between changing FC layers to conv layers
              and using FC layers directly. The results remain the same.
        """
        newlayers = []
        bias_switch = True
        first_linear = True
        h_w = tuple(x[0][0].size())

        for name, sequential_modules in self.model.named_children():
            if isinstance(sequential_modules, nn.AdaptiveAvgPool2d):
                h_w = sequential_modules.output_size
                newlayers += [sequential_modules]
            else:
                for module in sequential_modules:
                    if isinstance(module, nn.Conv2d):
                        h_w = conv_output_shape(h_w, module.kernel_size, module.stride,
                                                 module.padding, module.dilation)
                        channel_out = module.weight.shape[0]
                        newlayers += [module]
                    elif isinstance(module, nn.Linear):
                        newlayer = None
                        if module.bias is None: bias_switch = False

                        if first_linear == True:
                            n = module.weight.shape[0]
                            newlayer = nn.Conv2d(channel_out, n, h_w, bias=bias_switch)
                            newlayer.weight = nn.Parameter(module.weight.reshape(n, channel_out, h_w[0], h_w[1]))
                            first_linear = False
                        else:
                            m,n = module.weight.shape[1],module.weight.shape[0]
                            newlayer = nn.Conv2d(m,n,1, bias=bias_switch)
                            newlayer.weight = nn.Parameter(module.weight.reshape(n,m,1,1))

                        if bias_switch is not False: newlayer.bias = nn.Parameter(module.bias)

                        newlayers += [newlayer]

                    elif isinstance(module, nn.MaxPool2d) or isinstance(module, nn.AvgPool2d):
                        h_w = conv_output_shape(h_w, module.kernel_size, module.stride,
                                                 module.padding, module.dilation)                       
                        newlayers += [module]

                    elif isinstance(module, nn.Dropout):
                        newlayer = nn.Dropout2d(p = module.p)

                        newlayers += [newlayer]

                    else:
                        newlayers += [module]

        return newlayers 


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Compute output shape of convolutions
    """
    if type(h_w) is not tuple:
        h_w = (h_w, h_w)
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    
    if type(stride) is not tuple:
        stride = (stride, stride)
    
    if type(pad) is not tuple:
        pad = (pad, pad)

    if type(dilation) is not tuple:
        dilation = (dilation, dilation)
        
    h = (h_w[0] + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1)// stride[0] + 1
    w = (h_w[1] + (2 * pad[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1)// stride[1] + 1
    
    return h, w
   