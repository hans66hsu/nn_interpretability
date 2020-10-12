import torch
from pathlib import Path
import torchvision.datasets as datasets
from torchvision.transforms import transforms


class ImageNetValDataLoader:
    def __init__(self, batch_size: int = 64):
        self.valdir = Path(__file__).parent.parent.parent.joinpath('data/ImageNet/val')
        print(self.valdir)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        self.process = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self.valset = datasets.ImageFolder(str(self.valdir), self.process)
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, shuffle=True, num_workers=0)
