#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from theano import shared

FLOATX = theano.config.floatX


class Momentum(object):
    def __init__(self, params, momentum=0.9, epsilon=1e-10, nesterov=True):
        assert (momentum >= 0. and momentum < 1.)

        self._moments = [shared(value=np.zeros_like(p.get_value(), dtype=FLOATX))
                         for p in params]

        self.momentum = momentum
        self.epsilon = epsilon
        self.nesterov = nesterov

    def run_update(self, cost, params, learning_rate=0.01, consider_constants=None):
        """Function to compute the gradients of the cost w.r.t to the model parameters.

        Args:
            cost (TensorSharedVariable): the cost function to calculate the gradient
                from
            params (list()): a list of variables we want to
                calculate the gradient to
            learning_rate (float): learning rate used
            consider_constants (list(TensorSharedVariable)): a list of variables we do
                not backpropagate to

        Returns:
            list(theano.tensor.TensorVariable): list of gradient updates matching
                `params`
        """
        grads = [T.grad(cost, param, consider_constants) for param in params]

        moments = self._moments

        updates = []
        for m, param, g in zip(moments, params, grads):
            v = self.momentum * m - learning_rate * g  # velocity
            updates.append((m, v))

            if self.nesterov:
                update = param + self.momentum * v - learning_rate * g
            else:
                update = param + v

            updates.append((param, update))

        return updates
