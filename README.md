# CS231n 学习笔记

### CS231n: [Convolutional Neural Networks for Visual Recognition](http://vision.stanford.edu/teaching/cs231n/index.html)



## Course Description

计算机视觉已经在我们的社会中无处不在，应用于搜索，图像理解，应用程序，绘图，医学，无人驾驶飞机和自动驾驶汽车。许多这些应用的核心是视觉识别任务，例如图像分类，定位和检测。神经网络（又名“深度学习”）方法的最新发展极大地提高了这些最先进的视觉识别系统的性能。本课程深入探讨深度学习架构的细节，重点是学习这些任务的端到端模型，尤其是图像分类。在为期10周的课程中，学生将学习如何实施，训练和调试他们自己的神经网络，并详细了解计算机视觉的前沿研究。最终任务将涉及训练数百万参数卷积神经网络并将其应用于最大图像分类数据集（[ImageNet](http://image-net.org/challenges/LSVRC/2014/index)）。我们将专注于教授如何设置图像识别问题，学习算法（例如反向传播），培训和微调网络的实用工程技巧，并指导学生完成动手作业和最终课程项目。本课程的大部分背景和材料将来自ImageNet挑战赛。



## Notes & Videos

[CS231n官方笔记中文翻译](https://zhuanlan.zhihu.com/p/21930884)

[Course Notes](http://cs231n.github.io)

[Detailed Syllabus](http://vision.stanford.edu/teaching/cs231n/syllabus.html)

[2017 Lecture Videos (You Tube)](https://www.youtube.com/playlist?list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv)



## Course Notes

### 1 Spring 2018 Assignments

#### 1.1 Assignment #1:

[Image Classification, kNN, SVM, Softmax, Neural Network](http://cs231n.github.io/assignments2018/assignment1/)

#### 1.2 Assignment #2:

[Fully-Connected Nets, Batch Normalization, Dropout, Convolutional Nets](http://cs231n.github.io/assignments2018/assignment2/)

#### 1.3 Assignment #3:

[Image Captioning with Vanilla RNNs, Image Captioning with LSTMs, Network Visualization, Style Transfer, Generative Adversarial Networks](http://cs231n.github.io/assignments2018/assignment3/)

### 2 Module 0: Preparation

#### 2.1 [Setup Instructions](http://cs231n.github.io/setup-instructions/)

#### 2.2 [Python / Numpy Tutorial](http://cs231n.github.io/python-numpy-tutorial/)

#### 2.3 [IPython Notebook Tutorial](http://cs231n.github.io/ipython-tutorial/)

#### 2.4 [Google Cloud Tutorial](http://cs231n.github.io/gce-tutorial/)

#### 2.5 [AWS Tutorial](http://cs231n.github.io/aws-tutorial/)

### 3 Module 1: Neural Networks

#### 3.1 [Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits](http://cs231n.github.io/classification/)

*L1/L2 distances, hyperparameter search, cross-validation*

#### 3.2 [Linear classification: Support Vector Machine, Softmax](http://cs231n.github.io/linear-classify/)

*parameteric approach, bias trick, hinge loss, cross-entropy loss, L2 regularization, web demo*

#### 3.3 [Optimization: Stochastic Gradient Descent](http://cs231n.github.io/optimization-1/)

*optimization landscapes, local search, learning rate, analytic/numerical gradient*

#### 3.4 [Backpropagation, Intuitions](http://cs231n.github.io/optimization-2/)

*chain rule interpretation, real-valued circuits, patterns in gradient flow*

#### 3.5 [Neural Networks Part 1: Setting up the Architecture](http://cs231n.github.io/neural-networks-1/)

*model of a biological neuron, activation functions, neural net architecture, representational power*

#### 3.6 [Neural Networks Part 2: Setting up the Data and the Loss](http://cs231n.github.io/neural-networks-2/)

*preprocessing, weight initialization, batch normalization, regularization (L2/dropout), loss functions*

#### 3.7 [Neural Networks Part 3: Learning and Evaluation](http://cs231n.github.io/neural-networks-3/)

*gradient checks, sanity checks, babysitting the learning process, momentum (+nesterov), second-order methods, Adagrad/RMSprop, hyperparameter optimization, model ensembles*

#### 3.8 [Putting it together: Minimal Neural Network Case Study](http://cs231n.github.io/neural-networks-case-study/)

*minimal 2D toy data example*

### 4 Module 2: Convolutional Neural Networks

#### 4.1[Convolutional Neural Networks: Architectures, Convolution / Pooling Layers](http://cs231n.github.io/convolutional-networks/)

*layers, spatial arrangement, layer patterns, layer sizing patterns, AlexNet/ZFNet/VGGNet case studies, computational considerations*

#### 4.2 [Understanding and Visualizing Convolutional Neural Networks](http://cs231n.github.io/understanding-cnn/)

*tSNE embeddings, deconvnets, data gradients, fooling ConvNets, human comparisons*

#### 4.3 [Transfer Learning and Fine-tuning Convolutional Neural Networks](http://cs231n.github.io/transfer-learning/)



