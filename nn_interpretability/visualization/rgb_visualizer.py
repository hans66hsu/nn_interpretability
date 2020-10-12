import os 
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from torchvision.transforms import transforms
from PIL import Image
from nn_interpretability.visualization.lrp_heatmap import LRPHeatmap

path = None
mean = np.array([0.485, 0.456, 0.496])
std = np.array([0.229, 0.224, 0.225])


class RGBVisualizer(LRPHeatmap):
    @staticmethod
    def read_img(img_path, show = True):
        img = Image.open(img_path).convert('RGB')
        global path 
        path = img_path
        if show:
            plt.imshow(img)
        return img
    
    @staticmethod
    def preprocess(img):
        # Convert image from numpy to tensor
        to_tensor = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        return to_tensor(img).unsqueeze(0)

    @staticmethod
    def postprocess(img):
        # Convert image from tensor to numpy
        img = img.numpy().squeeze().transpose(1, 2, 0)
        img = img * std.reshape((1, 1, 3)) + mean.reshape((1, 1, 3))
        img = np.clip(img, 0.0, 255.0)
        return img

    @staticmethod 
    def save_img(img):
        global path
        img_path = os.path.splitext(path)[0]
        result = Image.fromarray((img * 255.).astype(np.uint8))
        result.save(str(img_path)+'_dream'+'.jpg')
    
    @staticmethod 
    def show(img, sx, sy):
        img = RGBVisualizer.postprocess(img)
        plt.figure(figsize=(sx, sy))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.imshow(img, interpolation='nearest')
        plt.show()
