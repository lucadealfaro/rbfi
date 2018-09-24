#!/usr/bin/env python

# Author: Luca de Alfaro, 2018
# License: BSD

from torch.nn import Linear
import torch
import numpy as np
from json_plus import Serializable

"""Implementation of a linear neural net with sensitivity analysis."""

class LinearWithSensitivity(Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearWithSensitivity, self).__init__(
            in_features, out_features, bias=bias
        )

    def overall_sensitivity(self):
        """Returns the sensitivity to adversarial examples of the layer."""
        if self.mod1:
            s = torch.max(torch.max(self.weight, -1)[0], -1)[0].item()
        else:
            s = torch.max(torch.sqrt(torch.sum(self.weight * self.weight, -1)))[0].item()
        s *= np.sqrt(2. / np.e)
        return s

    def sensitivity(self, previous_layer):
        """Given the sensitivity of the previous layer (a vector of length equal
        to the number of inputs), it computes the sensitivity to adversarial examples
         of the current layer, as a vector of length equal to the output size of the
         layer.  If the input sensitivity of the previous layer is None, then unit
         sensitivity is assumed."""
        if previous_layer is None:
            previous_layer = self.weight.new(1, self.in_features)
            previous_layer.fill_(1.)
        else:
            previous_layer = previous_layer.view(1, self.in_features)
        w = previous_layer * self.weight
        s = torch.sum(torch.abs(w), -1)
        return s

    def dumps(self):
        d = dict(
            in_features=self.in_features,
            out_features=self.out_features,
            weight=self.weight.data.cpu().numpy(),
            bias=None if self.bias is None else self.bias.data.cpu().numpy(),
        )
        return Serializable.dumps(d)

    @staticmethod
    def loads(s, device):
        d = Serializable.loads(s)
        m = LinearWithSensitivity(
            d['in_features'],
            d['out_features'],
            bias=d['bias'] is not None
        )
        m.weight.data = torch.from_numpy(d['weight']).to(device)
        if d['bias'] is not None:
            m.bias.data = torch.from_numpy(d['bias']).to(device)
        return m
