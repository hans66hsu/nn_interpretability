from torch.nn import Module
import torch.nn.functional as F
from torchvision.transforms import transforms

from nn_interpretability.interpretation.base.interpretation_base import DecisionInterpreter


class OcclusionSensitivity(DecisionInterpreter):
    """
    OcclusionSensitivity is a decision-based intepretability method which obstructs
    parts of the input in order to see what influence these regions have to the
    output of the model under test.
    """
    def __init__(self, model: Module, classes: [str], preprocess: transforms.Compose, input_size, block_size, fill_value, target):
        """
        :param model: The model the decisions of which needs to be interpreted.
        :param classes: A collection of all classes that the given model can classify
        :param preprocess: The preprocessing functions that need to be invoked for the model input.
        :param input_size: The expected 2D input size by the given model (e.g. (28, 28))
        :param block_size: The size of the 2D block which acts as occlusion.
                            This should be a divisor of each dimension of the input size! (e.g. (7, 7))
        :param fill_value: The value for the occlusion block
        :param target: The target class of the expected input
        """
        DecisionInterpreter.__init__(self, model, classes, preprocess)
        self.input_size = input_size
        self.block_size = block_size
        self.fill_value = fill_value
        self.target = target

        if self.input_size[0] % self.block_size[0] != 0 or self.input_size[1] % self.block_size[1] != 0:
            raise ValueError("The block size should be a divisor of the input size.")

    def _generate_occlused_input(self, x):
        out = []

        rows = int(self.input_size[0] / self.block_size[0])
        columns = int(self.input_size[1] / self.block_size[1])

        for row in range(rows):
            for column in range(columns):
                new_x = x.clone().detach().to(self.device)
                new_x[0][0][row * self.block_size[0]: (row + 1) * self.block_size[0], column * self.block_size[1]: (column + 1) * self.block_size[1]] = self.fill_value
                out.append(new_x)

        return out

    def _compute_probabilities(self, x):
        probabilities = []

        for i in range(len(x)):
            logits = self.model(x[i]).to(self.device)
            probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
            prob = round(probs[self.target], 2)
            probabilities.append(prob)

        return probabilities

    def interpret(self, x):
        x = self._execute_preprocess(x)

        occluded_input = self._generate_occlused_input(x)
        probabilities = self._compute_probabilities(occluded_input)

        occluded_input = [img.cpu() for img in occluded_input]

        return occluded_input, probabilities
