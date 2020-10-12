import torchvision
import torch
from torchvision.transforms import transforms
from scipy.ndimage.interpolation import rotate


class MnistDataLoader:
    def __init__(self, path: str = './data', batch_size: int = 64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        self.trainset = torchvision.datasets.MNIST(root=path, train=True, download=True, transform=self.transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)

        self.testset = torchvision.datasets.MNIST(root=path, train=False, download=True, transform=self.transform)
        self.testloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    def get_images_for_class(self, num_class):
        return self.trainset.data[self.trainset.targets == num_class]

    def get_image_for_class(self, num_class):
        # Get first instance of a certain class
        index = list(self.trainloader.dataset.targets).index(num_class)
        input, labels = self.trainloader.dataset[index]
        return input.reshape((1, 1, 28, 28)).to(self.device)

    def get_random_image(self):
        dataiter = iter(self.trainloader)
        images, labels = dataiter.next()

        image = images[0].expand(1, 1, 28, 28).to(self.device)
        label = labels[0].to(self.device)

        return image, label

    def generate_mean_image(self, images):
        image_sum = torch.zeros(images[0].size())

        for image in images:
            image_sum += image

        mean_image = (image_sum / len(images)).to(self.device)
        mean_image = (mean_image - mean_image.min()) / (mean_image.max() - mean_image.min())

        return mean_image.to(self.device)

    def generate_mean_image_for_class(self, num_class):
        images = self.get_images_for_class(num_class)

        return self.generate_mean_image(images).to(self.device)
    
    def rotate_image(self, num_class, deg):
        img = self.get_image_for_class(num_class).cpu()
        rotated_img = rotate(img.squeeze(0).numpy().transpose(1,2,0), deg, reshape=False)
        return torch.from_numpy(rotated_img).permute(2,0,1).unsqueeze_(0)