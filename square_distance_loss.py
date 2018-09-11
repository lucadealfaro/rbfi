#!/usr/bin/env python

# Author: Luca de Alfaro, 2018
# License: BSD

import torch

def square_distance_loss(output, target):
    """Returns the square distacne loss between output and target, except that
    the real reference output is a vector with all 0, with a 1 in the position
    specified by target."""
    s = list(output.shape)
    n_classes = s[-1]
    out = output.view(-1, n_classes)
    ss = out.shape
    n_els = ss[0]
    idxs = target.view(-1)
    t = output.new(n_els, n_classes)
    t.requires_grad=False
    t.fill_(0.)
    t[range(n_els), idxs] = 1.
    d = out - t
    dd = d * d
    return torch.sum(dd) / n_els
