#!/usr/bin/env python

# Author: Luca de Alfaro, 2018
# License: BSD

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd.function import Function

import numpy as np

from torch_bounded_parameters import BoundedParameter


class LargeAttractorExp(Function):
    """Implements e^-x with soft derivative."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(-x)

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_tensors[0]
        return - grad_output / torch.sqrt(1. + x)


class SharedFeedbackMax(Function):

    @staticmethod
    def forward(ctx, x):
        y, _ = torch.max(x, -1)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        y_complete = y.view(list(y.shape) + [1])
        d_complete = grad_output.view(list(grad_output.shape) + [1])
        return d_complete * torch.exp(x - y_complete)


class RBFI(nn.Module):

    def __init__(self, in_features, out_features, andor="*",
                 min_input=0.0, max_input=1.0, min_slope=0.001, max_slope=10.0):
        """
        Implementation of RBF module with logloss.
        :param in_features: Number of input features.
        :param out_features: Number of output features.
        :param andor: '^' for and, 'v' for or, '*' for mixed.
        :param min_input: minimum value for w (and therefore min value for input)
        :param max_input: max, as above.
        :param min_slope: min value for u, defining the slope.
        :param max_slope: max value for u, defining the slope.
        """
        super(RBFI, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.andor = andor
        self.w = BoundedParameter(torch.Tensor(out_features, in_features),
                                  lower_bound=min_input, upper_bound=max_input)
        self.u = BoundedParameter(torch.Tensor(out_features, in_features),
                           lower_bound=min_slope, upper_bound=max_slope)
        if andor == 'v':
            self.andor01 = Parameter(torch.ones((1, out_features)))
        elif andor == '^':
            self.andor01 = Parameter(torch.zeros((1, out_features,)))
        else:
            self.andor01 = Parameter(torch.Tensor(1, out_features,))
            self.andor01.data.random_(0, 2)
        self.andor01.requires_grad = False
        self.w.data.uniform_(min_input, max_input)
        # Initialization of u.
        self.u.data.uniform_(0.2, 0.7) # These could be parameters.
        self.u.data.clamp_(min_slope, max_slope)

    def forward(self, x):
        # Let n be the input size, and m the output size.
        # The tensor x is of shape * n. To make room for the output,
        # we view it as of shape * 1 n.
        s = list(x.shape)
        new_s = s[:-1] + [1, s[-1]]
        xx = x.view(*new_s)
        xuw = self.u * (xx - self.w)
        xuwsq = xuw * xuw
        # Aggregates into a modulus.
        z = SharedFeedbackMax.apply(xuwsq)
        y = LargeAttractorExp.apply(z)
        # Takes into account and-orness.
        if self.andor == '^':
            return y
        elif self.andor == 'v':
            return 1.0 - y
        else:
            return y + self.andor01 * (1.0 - 2.0 * y)

    def sensitivity(self, previous_layer):
        """Given the sensitivity of the previous layer (a vector of length equal
        to the number of inputs), it computes the sensitivity to adversarial examples
         of the current layer, as a vector of length equal to the output size of the
         layer.  If the input sensitivity of the previous layer is None, then unit
         sensitivity is assumed."""
        if previous_layer is None:
            previous_layer = self.w.new(1, self.in_features)
            previous_layer.fill_(1.)
        else:
            previous_layer = previous_layer.view(1, self.in_features)
        u = previous_layer * self.u
        s = torch.max(u, -1)[0]
        s *= np.sqrt(2. / np.e)
        return s
