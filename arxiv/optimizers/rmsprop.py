#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

from theano import shared

FLOATX = theano.config.floatX


class RMSProp(object):
    def __init__(self, params, rho=0.9, epsilon=1e-10, decay=0.):
        """Initialize the RMSProp optimizer"""
        assert (rho >= 0. and rho < 1.)
        assert (decay >= 0. and decay < 1.)

        self._accumulators = [shared(value=np.zeros_like(p.get_value(), dtype=FLOATX))
                              for p in params]

        self.rho = rho
        self.epsilon = epsilon
        self.decay = decay

    def run_update(self, cost, params, masks=None, learning_rate=0.001,
                   consider_constants=None):
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

        accumulators = self._accumulators

        updates = []
        for i, (a, param, g) in enumerate(zip(accumulators, params, grads)):
            # update accumulator

            new_a = self.rho * a + (1. - self.rho) * T.sqr(g)
            update = param - learning_rate * g / (T.sqrt(new_a) + self.epsilon)
            if masks is not None:
                if masks[i] is not None:
                    update = update * masks[i]
            updates.append((a, new_a))
            updates.append((param, update))

        return updates
