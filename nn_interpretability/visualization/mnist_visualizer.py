import numpy as np
import matplotlib.pyplot as plt

from nn_interpretability.visualization.lrp_heatmap import LRPHeatmap


class MnistVisualizer(LRPHeatmap):
    @staticmethod
    def show(image):
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

    @staticmethod
    def show_with_probability(image, probability):
        plt.title("Probability {:.2f} %".format(probability * 100))
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

    @staticmethod
    def show_examples(images):
        for i, img in enumerate(images):
            plt.figure(i)
            plt.imshow(img.reshape(28, 28), cmap='gray')

        plt.show()

    @staticmethod
    def show_dataset_examples(trainloader, n: int = 1):
        dataiter = iter(trainloader)
        images, _ = dataiter.next()

        images = images[0:n]
        MnistVisualizer.show_examples(images)

    @staticmethod
    def display_images_for_each_class(images):
        for i in range(5):
            plt.figure(figsize=(10, 10))
            for j in range(2):
                plt.subplot(5, 2, i * 2 + j + 1)
                plt.imshow(images[i * 2 + j], cmap='gray', vmin=0, vmax=1)
                plt.title('Digit {}'.format(i * 2 + j))
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()

    @staticmethod
    def display_images_with_probabilities_for_each_class(images, probabilities):
        for i in range(5):
            plt.figure(figsize=(10, 10))
            for j in range(2):
                plt.subplot(5, 2, i * 2 + j + 1)
                plt.imshow(images[i * 2 + j], cmap='gray', vmin=0, vmax=1)
                plt.title('Digit {}, Probability {:.2f}'.format(i * 2 + j, probabilities[i * 2 + j]))
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()

    @staticmethod
    def uncertain_deeplift_for_each_class(images, times):
        for i in range(5):
            plt.figure(figsize=(10, 10))
            for j in range(2):
                ax = plt.subplot(5, 2, i * 2 + j + 1)
                plt.imshow(images[i * 2 + j], cmap='gray', vmin=0, vmax=1)
                plt.text(0.12, -0.1, 'Predicted {:d} times'.format(int(times[i * 2 + j])),
                         transform=ax.transAxes)
                plt.title('Predicted as {}'.format(i * 2 + j))
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()

    @staticmethod
    def normalize(image):
        norm = (image-image.mean())/image.std()
        norm = norm * 0.1
        norm = norm + 0.5
        return norm.clamp(0, 1)

    @staticmethod
    def cat_images(images, sx, sy):
        plt.figure(figsize=(sx, sy))
        imgs_stack = np.hstack(image.squeeze(0).cpu().numpy().reshape(28, 28) for image in images)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.imshow(imgs_stack, interpolation='nearest', cmap='gray')
        plt.show()  

    @staticmethod 
    def result_for_each_rotation(images, cls, score, top_k, sx, sy):
        plt.figure(figsize=(sx, sy))
        for i in range(len(images)):
            ax = plt.subplot(1,len(images), i+1)
            for j in range(top_k):
                plt.text(0.5, -0.5-0.5*j, "Class: {} \n{:.2f} %"
                         .format(cls[i][j], score[i][j] * 100), size=12, 
                         ha="center", transform=ax.transAxes)
            plt.imshow(images[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.tight_layout()

    @staticmethod
    def display_images_with_probability(images, probabilities, sx, sy):
        plt.figure(figsize=(sx, sy))

        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.title("{:.2f} %".format(probabilities[i] * 100))
            plt.imshow(images[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.tight_layout()

    @staticmethod
    def display_images(images, sx, sy):
        plt.figure(figsize=(sx, sy))

        for i in range(len(images)):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(images[i].reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.tight_layout()
