# ResLogit
Residual Logit (ResLogit) models are a family of machine learning based fully interpretable choice models under-developement in the [Laboratory of Innovations in Transportation (LiTrans)](https://litrans.ca/). So far, two models are available for general consumption:
### 1. Standard ResLogit
Wong, Melvin, and Bilal Farooq. "[ResLogit: A residual neural network logit model for data-driven choice modelling.](https://doi.org/10.1016/j.trc.2021.103050)" Transportation Research Part C: Emerging Technologies 126 (2021): 103050. 
### 2. Ordinal Reslogit
Kamal, Kimia, and Bilal Farooq. "[Ordinal-ResLogit: Interpretable Deep Residual Neural Networks for Ordered Choices.](https://arxiv.org/abs/2204.09187)" arXiv preprint arXiv:2204.09187 (2022).

# Implementation
The original implementation of ResLogit was done using [Theano](https://github.com/Theano/Theano) in Python. However, as the development of Theano stopped, we have moved to [PyTorch](https://pytorch.org/) in Python as the active development platform. The old Theano version of the code can be found in [arxiv](https://github.com/LiTrans/reslogit/tree/master/arxiv).
