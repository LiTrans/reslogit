# ResLogit
Residual Logit (ResLogit) models are a family of machine learning based fully interpretable choice models underdevelopment in the [Laboratory of Innovations in Transportation (LiTrans)](https://litrans.ca/). So far, two models are available for general consumption:
### 1. Standard ResLogit
Wong, Melvin, and Bilal Farooq. "[ResLogit: A residual neural network logit model for data-driven choice modelling.](https://doi.org/10.1016/j.trc.2021.103050)" Transportation Research Part C: Emerging Technologies 126 (2021): 103050. 
### 2. Ordinal Reslogit
Kamal, Kimia, and Bilal Farooq. "[Ordinal-ResLogit: Interpretable Deep Residual Neural Networks for Ordered Choices.](https://www.sciencedirect.com/science/article/pii/S1755534523000556)" Journal of Choice Modelling 50 (2024).

# Implementation
The original implementation of ResLogit was done using [Theano](https://github.com/Theano/Theano) in Python. However, as the development of Theano stopped, we have moved to [PyTorch](https://pytorch.org/) in Python as the active development platform. The old Theano version of the code can be found in [arxiv](https://github.com/LiTrans/reslogit/tree/master/arxiv).

# Contributors
- [Kimia Kamal](https://github.com/kimiak91)
- [Melvin Wong](https://github.com/mwong009)
- [Bilal Farooq](https://github.com/billjee)
