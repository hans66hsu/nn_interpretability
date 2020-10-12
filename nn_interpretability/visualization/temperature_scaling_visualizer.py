import matplotlib.pyplot as plt
import numpy as np


class TemperatureScalingVisualizer:
    @staticmethod
    def bins(ax, data):
        weights = np.ones_like(data) / float(len(data))
        bins = [0.1 * i for i in range(1, 11)]
        ax.hist(data, bins=bins, weights=weights)

    @staticmethod
    def confidence_histogram(data):
        fig, ax = plt.subplots()

        TemperatureScalingVisualizer.bins(ax, data)

        ax.set_xlabel('Confidence')
        ax.set_ylabel('% of samples')
        ax.set_title('Confidence Histograms')
        fig.tight_layout()

    @staticmethod
    def reliability_diagram(data):
        fig, ax = plt.subplots()

        TemperatureScalingVisualizer.bins(ax, data)

        ax.set_xlabel('Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability diagram')

        fig.tight_layout()

