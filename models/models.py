import numpy as np
import pandas as pd
import timeit
import pickle
import torch
import torch.nn as nn
import yaml

Floatt = torch.float64

class Logit(object):
    def __init__(self, input, choice, n_vars, n_choices, beta=None, asc=None):
        """Initialize the Logit class.
        
        Args:
            input (TensorVariable)
            choice (TensorVariable)
            n_vars (int): number of input variables.
            n_choices (int): number of choice alternatives.
        """
        self.input = input
        self.choice = choice
        
        #define initial value for asc parameter and parameters associated to explanatory variables
        asc_init = torch.zeros((n_choices,), dtype = Floatt)
        if asc is None:
            asc = nn.Parameter(asc_init)
        self.asc = asc
        
        beta_init = torch.zeros((n_vars, n_choices), dtype = Floatt)
        if beta is None:
            beta = nn.Parameter(beta_init)
        self.beta = beta
        
        self.params = [self.beta, self.asc]
        
        #compute the utility function and the probability  of each alternative
        pre_softmax = torch.matmul(input, self.beta) + self.asc
        
        self.output = nn.functional.softmax(pre_softmax, dim =1)
        
        self.output_pred = torch.argmax(self.output, dim =1)
        

    def negative_log_likelihood_Ordinal(self, x, y):
        """Cost function: returns the sum of the negative log likelihood

        Args:
            y (TensorVariable) : ordinal level.
            x (TensorVariable) : the probability of ordinal alternatives.
        """
        
        self.cost = -torch.sum((torch.log(x)*(y))+(torch.log(1-x)*(1-y)))

        return self.cost
    
    
    def negative_log_likelihood(self, x, y):
        """Cost function: returns the sum of the negative log likelihood

        Args:
            y (TensorVariable) : the output.
            x (TensorVariable) : the probability of alternatives.
        """
        
        self.cost = -torch.sum(torch.log(x))[torch.arange(y.shape[0])]

        return self.cost
    
    def prob_choice(self,y):
        return self.output_logit
    
    def prediction(self):
        return self.output_pred
        
    def errors(self, y):
        """returns the number of errors in the minibatch for computing the accuracy of model.
        
         Args:
            y (TensorVariable):  the correct label. 
        """
        if y.ndim != self.output_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.output_pred',
                ('y', y.type, 'y_pred', self.output_pred_logit.type)
            )
        if y.dtype in [torch.int16, torch.int32, torch.int64]:
            not_equal = torch.ne(self.output_pred, y)
            return not_equal.sum().float() / not_equal.numel()
        else:
            raise NotImplementedError()


class ResNetLayer(nn.Module):
    def __init__(self, n_in, n_out):
        """Initialize the ResNetLayer class.
        
        Args:
            input (TensorVariable)
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
        """
        super(ResNetLayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        
        # define Initial value of residual layer weights
        W_init = torch.eye(self.n_out, dtype=Floatt)
        
        # learnable parameters of a model 
        self.W = nn.Parameter(W_init)
        self.params = [self.W]

    def forward(self, x):
        """ return the output of each residual layer.
        
         Args:
             x (TensorVariable):  input of each residual layer. 
        """
        self.lin_output = torch.matmul(x, self.W)

        output = x - nn.functional.softplus(self.lin_output)
        return output



class ResNet(nn.Module):
    def __init__(self, n_in, n_out, n_layers=16):
        """Initialize the ResNet architecture.

        Args:
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
            n_layers (int): number of residual layers.
        """
        super(ResNet, self).__init__()
        
        self.n_layers = n_layers
        
        #define n_layers residual layer
        self.layers = nn.ModuleList([ResNetLayer(n_in, n_out) for _ in range(n_layers)])
        
    def forward(self, x):
        """ return the final output of ResNet architecture.
        
         Args:
             x (TensorVariable):  input of first residual layer. 
        """
        out = x
        for i in range(self.n_layers):
            out = self.layers[i](out)
        return out


class Ordinal_layer(object):
    def __init__(self, n_choices):
        """Initialize the Ordinal_layer class (Coral layer).
        
        Args:
        n_choices (int): number of choice alternatives.
        """
        
        #define initial value for the parameters of ordinal layer 
        W_ordinal_init = torch.ones((n_choices, n_choices-1), dtype = Floatt)
        self.W_ordinal = nn.Parameter(W_ordinal_init)
    
        bias_ordinal_init = torch.ones((n_choices-1,), dtype = Floatt)
        self.bias_ordinal = nn.Parameter(bias_ordinal_init)
        
        self.params= [self.W_ordinal ,self.bias_ordinal]
        
        
    def fit_Ordinal_layer(self, input):
        """ return the output of Coral layer.
         
         Args:
             input (TensorVariable):  output of last residual layer. 
        """
        self.input = input
        
        ordinal_output = torch.matmul( self.input, self.W_ordinal) 
        
        self.logits = ordinal_output + self.bias_ordinal  


class OrdinalResLogit(Logit):
    def __init__(self, input, choice, n_vars, n_choices, n_layers=16, batch_size=128):
        """Initialize the OrdinalResLogit class.
        
        Args:
        input (TensorVariable)
        choice (TensorVariable) : ordinal level
        n_vars (int): number of input variables.
        n_choices (int): number of choice alternatives.
        n_layers (int): number of residual layers.
        batch_size (int): size of each batch.
        """
        Logit.__init__(self, input, choice, n_vars, n_choices)
        
        self.n_vars = n_vars
        self.n_choices = n_choices
        self.n_layers = n_layers
        self.batch_size = batch_size 
        
        #define the ResNet architecture.
        self.resnet_layer = ResNet(self.n_choices, self.n_choices, n_layers=16)
        for i in range(self.n_layers):
            self.params.extend(self.resnet_layer.layers[i].params)
            
        #define the ordinal layer.
        self.ordinal_layer = Ordinal_layer(self.n_choices)
        self.params.extend(self.ordinal_layer.params)
        
        
    def fit(self, input):

        self.input = input
        assert self.n_layers >= 1 
        
        resnet_input = torch.matmul(self.input, self.beta)

        output_resnet = self.resnet_layer.forward(resnet_input)
              
        pre_softmax = output_resnet + self.asc
        
        self.ordinal_layer.fit_Ordinal_layer(pre_softmax)
        
        self.output = self.ordinal_layer.logits
        self.output_likelihood = torch.sigmoid(self.ordinal_layer.logits)
        
        #prediction of each binary classifier
        for i in range(self.batch_size):      
            self.predict_levels =  self.output_likelihood > 0.5
            self.output_pred = torch.sum(self.predict_levels, dim=1) + 1
        
        
    def predict(self, input):
        
        self.input = input
        assert self.n_layers >= 1 
        
        resnet_input = torch.matmul(self.input, self.beta)

        output_resnet = self.resnet_layer.forward(resnet_input)
            
        #final output of resifual layers    
        pre_softmax = output_resnet + self.asc
        
        #ordinal layer
        self.ordinal_layer.fit_Ordinal_layer(pre_softmax)
        
        #output of ordinal layer
        self.output = self.ordinal_layer.logits
        self.output_likelihood = torch.sigmoid(self.ordinal_layer.logits)
        
        self.predict_levels =  self.output_likelihood > 0.5
        self.output_pred = torch.sum(self.predict_levels, dim=1) + 1
        
        return self.output_pred



class ResLogit(Logit):
    def __init__(self, input, choice, n_vars, n_choices, n_layers=16):
        """Initialize the ResLogit class.
        
        Args:
        input (TensorVariable)
        choice (TensorVariable) : actual label
        n_vars (int): number of input variables.
        n_choices (int): number of choice alternatives.
        n_layers (int): number of residual layers.
        """
        Logit.__init__(self, input, choice, n_vars, n_choices)
        
        self.n_vars = n_vars
        self.n_choices = n_choices
        self.n_layers = n_layers
        
        #define the ResNet architecture.
        self.resnet_layer = ResNet(self.n_choices, self.n_choices, n_layers=16)
        for i in range(self.n_layers):
            self.params.extend(self.resnet_layer.layers[i].params)
        
        
    def fit(self, input):
        
        self.input = input
        assert self.n_layers >= 1 
        
        resnet_input = torch.matmul(self.input, self.beta)

        output_resnet = self.resnet_layer.forward(resnet_input)
               
        pre_softmax = output_resnet + self.asc
        
        self.output = nn.functional.softmax(pre_softmax, dim =1)
        
        self.output_pred = torch.argmax(self.output, dim =1)