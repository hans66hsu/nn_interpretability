import torch
from torch.nn import Module
from torchvision import transforms


class Interpreter:
    """
    Implements an abstract Interpreter which serves as a base
    for any interpretation method.
    """

    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Module = model
        self.model.train()
        self.model = self.model.to(self.device)

        self.classes = classes
        self.preprocess = preprocess

        self._last_prediction = -1

    def _execute_preprocess(self, x):
        if self.preprocess:
            x = self.preprocess(x)

        return x.to(self.device)

    def _one_hot(self, output, class_idx):
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        return one_hot.to(self.device)

    def normalize(self, x):
        return ((x - x.min()) / (x.max() - x.min())).to(self.device)

    def last_prediction(self):
        return self._last_prediction


class ModelInterpreter(Interpreter):
    """
    Implements an abstract ModelInterpreter which serves as a base
    for any model-based interpretation method.
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """
        Interpreter.__init__(self, model, classes, preprocess)

    def interpret(self):
        """
        Provides an interpretation for the specified model.
        """
        pass


class DecisionInterpreter(Interpreter):
    """
    Implements an abstract DecisionInterpreter which serves as a base
    for any decision-based interpretation method.
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        """
        Interpreter.__init__(self, model, classes, preprocess)

    def interpret(self, x):
        """
        Provides an interpretation for the decision of the specified model for the defined input.

        :param x: The input which will be fed into the specified model.
        :return:
        """
        pass
