#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T


class SGD(object):
    def __init__(self, params):
        """Initialize the stochastic gradient descent algorithm"""
        pass

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
        self.grads = grads

        updates = []
        for param, g in zip(params, grads):
            update = param - learning_rate * g
            updates.append((param, update))

        return updates
