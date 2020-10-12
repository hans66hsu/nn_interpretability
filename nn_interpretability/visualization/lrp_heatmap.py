import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class LRPHeatmap:
    @staticmethod
    def heatmap(R, sx, sy):
        b = 10*((np.abs(R)**3.0).mean()**(1.0/3))
        my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
        my_cmap[:, 0:3] *= 0.85
        my_cmap = ListedColormap(my_cmap)
        plt.figure(figsize=(sx, sy))
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        plt.axis('off')
        plt.imshow(R, cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
        plt.show()

    @staticmethod
    def display_heatmap_for_each_class(images):
        for i in range(5):
            plt.figure(figsize=(10, 10))
            for j in range(2):
                b = 10*((np.abs(images[i * 2 + j])**3.0).mean()**(1.0/3))
                my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
                my_cmap[:,0:3] *= 0.85
                my_cmap = ListedColormap(my_cmap)
                plt.subplot(5, 2, i * 2 + j + 1)
                plt.imshow(images[i * 2 + j], cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
                plt.title('Digit {}'.format(i * 2 + j))
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
            
    @staticmethod             
    def dropout_heatmap_for_each_class(images, times):
        for i in range(5):
            plt.figure(figsize=(10, 10))
            for j in range(2):
                b = 10*((np.abs(images[i * 2 + j].sum(axis=0))**3.0).mean()**(1.0/3))
                my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
                my_cmap[:,0:3] *= 0.85
                my_cmap = ListedColormap(my_cmap)
                ax = plt.subplot(5, 2, i * 2 + j + 1)
                plt.imshow(images[i * 2 + j].sum(axis=0), cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
                plt.text(0.12, -0.1, 'Predicted {:d} times'.format(int(times[i * 2 + j])),
                         transform=ax.transAxes)
                plt.title('Predicted as {}'.format(i * 2 + j))
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
    
    @staticmethod
    def edl_heatmap_for_each_rotation(images, predicted, uncertainty):
        for i in range(2):
            plt.figure(figsize=(10, 10))
            for j in range(5):
                b = 10*((np.abs(images[i * 5 + j])**3.0).mean()**(1.0/3))
                my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
                my_cmap[:,0:3] *= 0.85
                my_cmap = ListedColormap(my_cmap)
                ax = plt.subplot(2, 5, i * 5 + j + 1)
                plt.imshow(images[i * 5 + j], cmap=my_cmap, vmin=-b, vmax=b, interpolation='nearest')
                if uncertainty is not None: 
                    plt.text(0.12, -0.1, 'Uncertainty {:.1f}% '.format(uncertainty[i * 5 + j] * 100),
                             transform=ax.transAxes)
                plt.title('Predicted as {}'.format(int(predicted[i * 5 + j])))
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
