# Interpretability of Neural Networks

## Project overview

Pytorch implementation of various neural network interpretability methods and how they can interpret uncertainty awareness models. 

The main implementation can be found in the `nn_interpretability` package. We also provide every method an accompanied Jupyter Notebook to demonstrate how we can use the `nn_interpretability` package in practice. Some of the methods are showcased together in one notebook for better comparison. Next to the interpretability functionality, we have defined a repository for models we trained and additional functionality for loading and visualizing data and the results from the interpretability methods. Furthermore, we have implemented uncertainty techniques to observe the behavior of interpretability methods under stochastical settings.

## Setup
The main deliverable of this repository is the package `nn_interpretability`, which entails every implementation of a NN interpretability method that we have done as part of the course. It can be installed and used as a library in any project. In order to install it one should clone this repository and execute the following command:
```
pip install -e .
```
After that, the package can be used anywhere by importing it:
```
import nn_interpretability as nni
```
An example usage of a particular interpretability method can be found in the corresponding Jupyter Notebook as outlined below. We also prepared a general demonstration of the developed package in this [Jupyter Notebook.](./14.Demo.ipynb)

 * Note: The package assume that layers of the model are constructed inside containers(e.g. features and classifier). This setting is due to the structure of the pretrained models from the model zoo. You could use `torch.nn.Sequential` or `torch.nn.ModuleList` to achieve this on your own model.

## IV. Table of neural network interpretability methods
### 1. Model-based approaches
 - Activation Maximization
   - [X] General Activation Maximization | [Jupyter Notebook](./1.Activation_Maximization.ipynb)
   - [X] Activation Maximization in Codespace (GAN) | [Jupyter Notebook](./1.Activation_Maximization.ipynb)
   - [X] Activation Maximization in Codespace (DCGAN) | [Jupyter Notebook](./1.Activation_Maximization.ipynb)
 - DeepDream | [Jupyter Notebook](./2.Deep_Dream.ipynb)

### 2. Decision-based approaches
 - Saliency Map | [Jupyter Notebook](./3.Saliency_Maps.ipynb)
 - DeConvNet
   - [X] Full Input Reconstruction | [Jupyter Notebook](./4.Deconvolution.ipynb)
   - [X] Partial Input Reconstruction | [Jupyter Notebook](./4.Deconvolution.ipynb)
 - Occlusion Sensitivity | [Jupyter Notebook](./5.Occlusion_Sensitivity.ipynb)
 - Backpropagation
   - [X] Vallina Backpropagation | [Jupyter Notebook](./6.Backpropagation.ipynb)
   - [X] Guided Backpropagation | [Jupyter Notebook](./6.Backpropagation.ipynb)
   - [X] Integrated Gradients | [Jupyter Notebook](./6.Backpropagation.ipynb)
   - [X] SmoothGrad | [Jupyter Notebook](./6.Backpropagation.ipynb)
 - Taylor Decomposition
   - [X] Simple Taylor Decomposition | [Jupyter Notebook](./7.Taylor_Decomposition.ipynb)
   - [X] Deep Taylor Decomposition | [Jupyter Notebook](./7.Taylor_Decomposition.ipynb)
 - LRP
   - [X] LRP-0 | [Jupyter Notebook](./8.1.LRP.ipynb)
   - [X] LRP-epsilon | [Jupyter Notebook](./8.1.LRP.ipynb)
   - [X] LRP-gamma | [Jupyter Notebook](./8.1.LRP.ipynb) 
   - [X] LRP-ab | [Jupyter Notebook](./8.1.LRP.ipynb)
 - DeepLIFT
   - [X] DeepLIFT Rescale | [Jupyter Notebook](./9.DeepLIFT.ipynb)
   - [X] DeepLIFT Linear | [Jupyter Notebook](./9.DeepLIFT.ipynb)
   - [X] DeepLIFT RevealCancel | [Jupyter Notebook](./9.DeepLIFT.ipynb)
 - CAM
   - [X] Class Activation Map (CAM) | [Jupyter Notebook](./10.1.Class_Activation_Map.ipynb)
   - [X] Gradient-Weighted Class Activation Map (Grad-CAM) | [Jupyter Notebook](./10.2.Grad_Class_Activation_Map.ipynb)

### 3. Uncertainty
 - Monte Carlo Dropout 
   - [X] Monte Carlo Dropout Analysis | [Jupyter Notebook](./11.MC_Dropout_Interpretability.ipynb)
   - [X] Uncertainty interpretability with LRP | [Jupyter Notebook](./11.MC_Dropout_Interpretability.ipynb)
 - Evidential Deep Learning
   - [X] Evidential Deep Learning Anaylsis | [Jupyter Notebook](./12.Evidential_Interpretability.ipynb)
   - [X] Base Model vs. Evidential Deep Learning Model with LRP | [Jupyter Notebook](./12.Evidential_Interpretability.ipynb)
 - Uncertain DeepLIFT
   - [X] DeepLIFT Deterministic vs. Stochastic Model | [Jupyter Notebook](./13.Uncertainty_Aware_DeepLIFT.ipynb)
   - [X] DeepLIFT Random Noise | [Jupyter Notebook](./13.Uncertainty_Aware_DeepLIFT.ipynb)
   - [X] Temperature scaling | [Jupyter Notebook](./13.Uncertainty_Aware_DeepLIFT.ipynb)

## References
- [X] Methods for Interpreting and Understanding Deep Neural Networks | [Paper](https://arxiv.org/abs/1706.07979) | [Implementation I](./1.Activation_Maximization.ipynb) |  [Implementation II](./8.1.LRP.ipynb) | [Implementation III](./8.2.LRP_Transpose.ipynb)
- [X] Feature Visualization | [Paper](https://distill.pub/2017/feature-visualization/)
- [X] The Building Blocks of Interpretability | [Paper](https://distill.pub/2018/building-blocks/)
- [X] Visualising image classification models and saliency maps | [Paper](https://arxiv.org/pdf/1312.6034.pdf) | [Implementation](./3.Saliency_Maps.ipynb)
- [X] Visualizing and understanding convolutional networks | [Paper](https://arxiv.org/pdf/1311.2901.pdf) | [Implementation](./4.Deconvolution.ipynb) | [Implementation III](./5.Occlusion_Sensitivity.ipynb)
- [X] Striving for Simplicity: The All Convolutional Net | [Paper](https://arxiv.org/pdf/1412.6806.pdf) | [Implementation](./6.Backpropagation.ipynb)
- [X] Axiomatic Attribution for Deep Networks | [Paper](https://arxiv.org/pdf/1703.01365.pdf) | [Implementation](./6.Backpropagation.ipynb)
- [X] Layer-Wise Relevance Propagation: An Overview | [Paper](http://iphome.hhi.de/samek/pdf/MonXAI19.pdf) | [Implementation](./8.1.LRP.ipynb)
- [X] SmoothGrad: removing noise by adding noise | [Paper](https://arxiv.org/pdf/1706.03825.pdf) | [Implementation](./6.Backpropagation.ipynb)
- [X] Learning Deep Features for Discriminative Localization | [Paper](http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf) | [Implementation](./10.1.Class_Activation_Map.ipynb)
- [X] Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization | [Paper](https://arxiv.org/pdf/1610.02391.pdf) | [Implementation](./10.2.Grad_Class_Activation_Map.ipynb)
- [X] Learning Important Features Through Propagating Activation Differences
 | [Paper](https://arxiv.org/pdf/1704.02685.pdf) | [Implementation](./9.DeepLIFT.ipynb)
- [X] Inceptionism: Going Deeper into Neural Networks | [Paper](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html) | [Implementation](./2.Deep_Dream.ipynb)
- [X] On Calibration of Modern Neural Networks
 | [Paper](http://proceedings.mlr.press/v70/guo17a/guo17a.pdf) | [Implementation](./13.Uncertainty_Aware_DeepLIFT.ipynb)

