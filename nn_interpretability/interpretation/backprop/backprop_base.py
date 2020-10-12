from torch.nn import Module
from torchvision.transforms import transforms

from nn_interpretability.interpretation.base.interpretation_base import ModelInterpreter


class BackPropBase(ModelInterpreter):
    """
    Base class for Backpropagation interpretability methods.
    """    
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """        
        super(BackPropBase, self).__init__(model, classes, preprocess)
        self.gradient = None
    
    def generate_gradient(self, input_model, x):
        x = (x.detach().to(self.device)).requires_grad_(True) 
        scores = input_model(x)

        #Clear gradients
        input_model.zero_grad()

        score_max_index = scores.argmax().item()
        score_max = scores[0,score_max_index]
        score_max.backward()
        
        return x.grad.detach().cpu()
