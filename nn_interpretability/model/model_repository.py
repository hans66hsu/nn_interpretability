import os
import torch
from pathlib import Path

from nn_interpretability.model.definition.am_mnist_classifier import AMCNN
from nn_interpretability.model.definition.mc_dropout_cnn import CNN_Dropout
from nn_interpretability.model.definition.general_mnist_cnn import GeneralCNN
from nn_interpretability.model.definition.mnist_generator import MNISTGenerator
from nn_interpretability.model.definition.mnist_discriminator import MNISTDiscriminator
from nn_interpretability.model.definition.cam_mnist_classifier import CAMMNISTClassifier
from nn_interpretability.model.definition.pretrained_dc_generator import PretrainedDCGANGenerator
from nn_interpretability.model.definition.cam_mnist_classifier_2 import CAMMNISTExtendedClassifier


class ModelRepository:
    MODELS_PATH = str(Path(__file__).parent.parent.parent.joinpath('models')) + "/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_general_mnist_cnn(path: str = None):
        model = GeneralCNN()

        if path is not None:
            if os.path.exists(ModelRepository.MODELS_PATH + path):
                model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)

    @staticmethod
    def get_cnn_dropout(path: str = None):
        model = CNN_Dropout()

        if path is not None:
            if os.path.exists(ModelRepository.MODELS_PATH + path):
                model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)
    
    @staticmethod
    def get_cam_classifier(path: str = None):
        model = CAMMNISTClassifier()

        if path is not None:
            model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)

    @staticmethod
    def get_cam_extended_classifier(path: str = None):
        model = CAMMNISTExtendedClassifier()

        if path is not None:
            model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)

    @staticmethod
    def get_am_classifier(path: str = None):
        model = AMCNN()

        if path is not None:
            model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)

    @staticmethod
    def get_pretrained_dcgan_generator():
        """
        Source of the pretrained model is:

        https://github.com/csinva/gan-vae-pretrained-pytorch
        :return:
        """
        path = 'pretrained_dcgan_generator.pth'

        model = PretrainedDCGANGenerator()
        model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)

    @staticmethod
    def get_mnist_generator(latent_dim: int = 128, path: str = None):
        model = MNISTGenerator(latent_dim=latent_dim)

        if path is not None:
            model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)

    @staticmethod
    def get_mnist_discriminator(path: str = None):
        model = MNISTDiscriminator()

        if path is not None:
            model = ModelRepository._load(model, path)

        return model.to(ModelRepository.device)

    @staticmethod
    def save(model, model_name):
        torch.save(model.state_dict(), ModelRepository.MODELS_PATH + model_name)
        return model

    @staticmethod
    def _load(model, model_name):
        model.load_state_dict(torch.load(ModelRepository.MODELS_PATH + model_name,  map_location=ModelRepository.device))
        return model.to(ModelRepository.device)
