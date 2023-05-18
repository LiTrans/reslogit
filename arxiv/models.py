import numpy as np
import pandas as pd
import theano
import theano.tensor as T

from theano import shared

__license__ = "MIT"
__revision__ = "2019-06-18 16:48:21"
__docformat__ = 'reStructuredText'

FLOATX = theano.config.floatX


class ResNetLayer(object):
    def __init__(self, input, n_in, n_out, scale):
        """Wrapper class object for ResNet layers

        Args:
            input (theano.tensor.TensorVariable): symbolic variable that describes
                the input.
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
            W (TensorSharedVariable, optional): hidden layer parameter.
            bias (TensorSharedVariable, optional): hidden layer parameter.
        """
        self.input = input

        W_init = np.diag(np.ones(n_out, dtype=FLOATX))
        W = shared(
            value=W_init,
            name='W',
            borrow=True
        )
        self.W = W

        W_mask = np.asarray(1.-np.diag([1. for i in range(n_out)]), dtype=FLOATX)

        bias_init = np.zeros((n_out,), dtype=FLOATX)
        bias = shared(
            value=bias_init,
            name='bias',
            borrow=True,
        )
        self.bias = bias

        W_init_2 = np.diag(np.ones(n_out, dtype=FLOATX))
        W_2 = shared(
            value=W_init_2,
            name='W_2',
            borrow=True
        )
        self.W_2 = W_2

        W_mask_2 = np.asarray(1.-np.diag([1. for i in range(n_out)]), dtype=FLOATX)

        self.params = [self.W]

        self.params_mask = [None]

        lin_output = T.dot(self.input, self.W)
        # lin_output = T.dot(T.nnet.softplus(lin_output) + self.bias, self.W_2)

        self.output = self.input - T.nnet.softplus(lin_output)


class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W=None, bias=None, activation=None):
        """Wrapper class object for intermediate layers

        Args:
            input (theano.tensor.TensorVariable): symbolic variable that describes
                the input.
            n_in (int): dimensionality of input.
            n_out (int): dimensionality of output.
            W (TensorSharedVariable, optional): hidden layer parameter.
            bias (TensorSharedVariable, optional): [description]. Defaults to None.
            activation (Elemwise, optional): symbolic non-linearity function
        """
        self.input = input

        if W is None:
            W = shared(
                value=np.zeros((n_in, n_out), dtype=FLOATX),
                name='W',
                borrow=True,
            )
        self.W = W

        if bias is None:
            bias = shared(
                value=np.zeros((n_out,), dtype=FLOATX),
                name='bias',
                borrow=True,
            )
        self.bias = bias

        # self.params = [self.W, self.bias]
        # lin_output = T.dot(input, self.W) + self.bias

        self.params = [self.W]
        lin_output = T.dot(input, self.W)

        if activation is None:
            self.output = lin_output
        else:
            self.output = activation(lin_output)


class Logit(object):
    def __init__(self, input, choice, n_vars, n_choices, beta=None, asc=None):
        """Initialize the Logit class.

        Args:
            input (theano.tensor.TensorVariable): symbolic variable that describes
                the input.
            choice (theano.tensor.TensorVariable): symbolic variable that describes
                the output.
            n_vars (int): number of input variables.
            n_choices (int): number of choice alternatives.
        """
        self.input = input
        self.choice = choice

        asc_init = np.zeros((n_choices,), dtype=FLOATX)
        if asc is None:
            asc = shared(
                value=asc_init,
                name='asc',
                borrow=True,
            )
        self.asc = asc

        beta_init = np.zeros((n_vars,n_choices), dtype=FLOATX)
        if beta is None:
            beta_flat = shared(
                value=beta_init.flatten(),
                name='beta_flat',
                borrow=True,
            )
            beta = beta_flat.reshape((n_vars, n_choices))
            beta.name = 'beta'
            
        self.beta_flat = beta_flat
        self.beta = beta

        self.params = [self.beta_flat, self.asc]

        beta_mask = np.ones_like(beta_init)
        beta_mask[..., -1] = 0
        beta_mask = beta_mask.flatten()

        asc_mask = np.ones_like(asc_init)
        asc_mask[..., -1] = 0

        self.params_mask = [beta_mask, asc_mask]

        pre_softmax = T.dot(input, self.beta) + self.asc

        self.output = T.nnet.softmax(pre_softmax)

        self.output_pred = T.argmax(self.output, axis=1)

    def negative_log_likelihood(self, y):
        """Returns the sum of the negative log likelihood

        Args:
            y (theano.tensor.TensorVariable): symbolic variable that describes
                the output.
        """

        # y.shape[0] is (symbolically) the number of rows in y
        return -T.sum(T.log(self.output)[T.arange(y.shape[0]), y])
    
    def prob_choice(self, y):
        
        return self.output

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch

        Args:
            y (theano.tensor.TensorVariable): corresponds to a vector that gives for
                each example the correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.output_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.output_pred',
                ('y', y.type, 'y_pred', self.output_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.output_pred, y))
        else:
            raise NotImplementedError()

    def get_gessians(self, y):
        """Return a list of hessians wrt to the model parameters

        Args:
            y (theano.tensor.TensorVariable): corresponds to a vector that gives for
                each example the correct label.

        Returns:
            list(TensorSharedVariable): a list of hessian matrix
        """
        hessians = []
        for param in [self.beta_flat, self.asc]:
            shp = param.shape
            batch_size = y.shape[0]
            cost = self.negative_log_likelihood(y)
            h = T.hessian(cost, param, disconnected_inputs='ignore')
            hessians.append(h)
        
        return hessians
    
    def get_derivatives(self, y):
        derivative = []
        for param in [self.beta_flat, self.asc]:
            shp = param.shape
            batch_size = y.shape[0]
            cost = self.negative_log_likelihood(y)
            d = T.grad(cost, param, disconnected_inputs='ignore')
            derivative.append(d)
        
        return derivative


class ResNet(Logit):
    def __init__(self, input, choice, n_vars, n_choices, n_layers=1):
        """Initialize the ResNet Model

        Args:
            input (theano.tensor.TensorVariable): symbolic variable that describes
                the input.
            choice (theano.tensor.TensorVariable): symbolic variable that describes
                the output.
            n_vars (int): number of input variables.
            n_choices (int): number of choice alternatives.
            layers (int, optional): list of ints. len() corresponds to the number
                of layers and the number of neurons (int) in each layer. Each layer
                size must equal to the preceding layer output size.
        """
        Logit.__init__(self, input, choice, n_vars, n_choices)

        self.resnet_layers = []

        self.n_layers = n_layers
        assert self.n_layers >= 1

        resnet_input = T.dot(input, self.beta)

        for i in range(self.n_layers):
            if i == 0:
                layer_input = resnet_input
            else:
                layer_input = self.resnet_layers[-1].output

            resnet_layer = ResNetLayer(layer_input, n_choices, n_choices, scale=i)
            self.resnet_layers.append(resnet_layer)
            self.params.extend(resnet_layer.params)
            self.params_mask.extend(resnet_layer.params_mask)

        pre_softmax = self.resnet_layers[-1].output + self.asc

        self.output = T.nnet.softmax(pre_softmax)

        self.output_pred = T.argmax(self.output, axis=1)


class MLP(Logit):
    def __init__(self, input, choice, n_vars, n_choices, n_layers=1):
        """Initialize the MLP class.

        Args:
            input (theano.tensor.TensorVariable): symbolic variable that describes
                the input.
            choice (theano.tensor.TensorVariable): symbolic variable that describes
                the output.
            n_vars (int): number of input variables.
            n_choices (int): number of choice alternatives.
        """
        Logit.__init__(self, input, choice, n_vars, n_choices)

        self.params = [self.asc]
        self.hidden_layers = []

        self.n_layers = n_layers
        assert self.n_layers >= 1

        for i in range(self.n_layers):
            if i == 0:
                layer_input = input
                input_size = n_vars
            else:
                layer_input = self.hidden_layers[-1].output
                input_size = n_choices

            hidden_layer = HiddenLayer(
                input=layer_input,
                n_in=input_size,
                n_out=n_choices,
                activation=T.nnet.softplus
            )

            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)

        pre_softmax = self.hidden_layers[-1].output + self.asc

        self.output = T.nnet.softmax(pre_softmax)

        self.output_pred = T.argmax(self.output, axis=1)
