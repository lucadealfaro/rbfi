#!/usr/bin/env python

# Author: Luca de Alfaro, 2018
# License: BSD

# Permutation-invariant MNIST.

from __future__ import print_function
import argparse
import json

import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from rbfi_pytorch import RBFI
from square_distance_loss import square_distance_loss
from torch_bounded_parameters import ParamBoundEnforcer


class RBFNet(nn.Module):
    """RBF Neural net."""

    def __init__(self, args):
        super(RBFNet, self).__init__()
        self.args = args
        layer_sizes = map(int, args.layers.split(',')) + [10] # 10 for the output.
        self.layers = []
        previous_size = 28 * 28
        for i, size in enumerate(layer_sizes):
            l = RBFI(previous_size, size,
                     andor=args.andor[i],
                     min_slope=args.min_slope,
                     max_slope=args.max_slope)
            self.layers.append(l)
            self.add_module('layer_%d' % i, l)
            previous_size = size

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def sensitivity(self):
        s = None
        for l in self.layers:
            s = l.sensitivity(s)
        return torch.max(s)

def train_once(args, model, flat_data, target, meta_optimizer):
    # For RBF, the optimizer also enforces parameter bounds.
    meta_optimizer.optimizer.zero_grad()
    output = model(flat_data)
    loss = square_distance_loss(output, target) + args.sensitivity_cost * model.sensitivity()
    loss.backward()
    meta_optimizer.optimizer.step()
    meta_optimizer.enforce()
    return output, loss

def train(args, model, device, train_loader, meta_optimizer, epoch):
    model.train()
    correct = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        flat_data = data.view(-1, 28 * 28)
        output, loss = train_once(args, model, flat_data, target, meta_optimizer)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAccuracy:{:.5f} Sensitivity: {}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss.item(),
                100. * correct / float(args.log_interval * args.batch_size),
                model.sensitivity().item()
            ))
            correct = 0.

def test(args, model, device, test_loader):
    model.eval()
    correct = 0.
    confidences = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            flat_data = data.view(-1, 28 * 28)
            output = model(flat_data)
            _, pred = output.max(1, keepdim=True)
            confidence = output.max(1)[0] / torch.sum(output, 1)
            correct += pred.eq(target.view_as(pred)).sum().item()
            confidences.append(torch.mean(confidence))
    accuracy = 100. * correct / len(test_loader.dataset)
    sensitivity = float(model.sensitivity().item())
    print('\nTest set: Accuracy: {}/{} ({:.5f}%, Sensitivity: {} Confidence: {})\n'.format(
        correct, len(test_loader.dataset),
        accuracy,
        sensitivity,
        np.mean(confidences)
    ))
    return accuracy, sensitivity


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--layers', type=str, default="200,40",
                        help='comma-separated list of layer sizes')
    parser.add_argument('--andor', type=str, default='**v',
                        help='Type of neurons in RBF nets, ^ = and, v = or, * = random mix')
    parser.add_argument('--min_slope', type=float, default=0.01,
                        help='Minimum slope for RBF (default: 0.01)')
    parser.add_argument('--max_slope', type=float, default=3.0,
                        help='Maximum slope for RBF (default: 3.0)')
    parser.add_argument('--sensitivity_cost', type=float, default=0.,
                        help='Cost of sensitivity')
    parser.add_argument('--opt', type=str, default='ada',
                        help='"ada" for adadelta, "mom" for momentum')
    parser.add_argument('--lr', type=float, default=5., metavar='LR',
                        help='learning rate (default: 5 for RBF)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs to perform')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.runs == 1:
        # We listen to the seed only for single runs.
        torch.manual_seed(args.seed)

    for run_idx in range(args.runs):
        print("====== Run {} ======".format(run_idx))
        print(json.dumps(args.__dict__))
        if args.runs > 1:
            torch.manual_seed(run_idx)
        # Creates the model.
        model = RBFNet(args).to(device)
        # Creates an optimizer.
        params = filter(lambda p: p.requires_grad, model.parameters())
        if args.opt == 'mom':
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.Adadelta(params, lr=args.lr)
        # For RBF nets, we wrap it into an enforcer.
        meta_optimizer = ParamBoundEnforcer(optimizer)
        # Trains the model for given number of epochs.
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, meta_optimizer, epoch)
        test(args, model, device, test_loader)


if __name__ == '__main__':
    main()