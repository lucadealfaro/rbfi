#!/usr/bin/env python

# Author: Luca de Alfaro, 2018
# License: BSD

# Permutation-invariant MNIST.

from __future__ import print_function
import argparse
import datetime
import json

import torch
from torch import nn
from nn_linear_pytorch_sensitivity import LinearWithSensitivity
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from rbf_pseudoderivative_pytorch import RBFPseudoDerivativeLayer
from square_distance_loss import square_distance_loss
from torch_bounded_parameters import ParamBoundEnforcer

# Loss functions
loss_relu = F.nll_loss
loss_sigmoid = square_distance_loss


class StdNet(nn.Module):
    """Standard neural net, which can be sigmoid or RELU, depending on flags."""
    def __init__(self, args):
        super(StdNet, self).__init__()
        self.args = args
        layer_sizes = map(int, args.layers.split(',')) + [10] # 10 for the output.
        self.layers = []
        previous_size = 28 * 28
        for i, size in enumerate(layer_sizes):
            l = LinearWithSensitivity(previous_size, size)
            self.layers.append(l)
            self.add_module('layer_%d' % i, l)
            previous_size = size

    def forward(self, x):
        for i, l in enumerate(self.layers):
            if self.args.sigmoid:
                x = F.sigmoid(l(x))
            else:
                # For RELU nets, the last layer is a softmax layer.
                if i < len(self.layers) - 1:
                    x = F.relu(l(x))
                else:
                    x = F.log_softmax(l(x), dim=1)
        return x

    def sensitivity(self):
        s = None
        for l in self.layers:
            s = l.sensitivity(s)
        return torch.max(s)


class RBFNet(nn.Module):
    """RBF Neural net."""

    def __init__(self, args):
        super(RBFNet, self).__init__()
        self.args = args
        layer_sizes = map(int, args.layers.split(',')) + [10] # 10 for the output.
        self.layers = []
        previous_size = 28 * 28
        for i, size in enumerate(layer_sizes):
            l = RBFPseudoDerivativeLayer(previous_size, size,
                                         andor=args.andor[i],
                                         modinf=args.modinf,
                                         min_slope=args.min_slope,
                                         max_slope=args.max_slope,
                                         regular_deriv=args.regular_deriv)
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
    if args.rbf:
        # For RBF, the optimizer also enforces parameter bounds.
        meta_optimizer.optimizer.zero_grad()
        output = model(flat_data)
        loss = square_distance_loss(output, target) + args.sensitivity_cost * model.sensitivity()
        loss.backward()
        meta_optimizer.optimizer.step()
        meta_optimizer.enforce()
    else:
        # Otherwise, it's standard as in pytorch.
        meta_optimizer.zero_grad()
        output = model(flat_data)
        model_loss = loss_sigmoid(output, target) if args.sigmoid else loss_relu(output, target)
        loss = model_loss + args.sensitivity_cost * model.sensitivity()
        loss.backward()
        meta_optimizer.step()
    return output, loss

def train(args, model, device, train_loader, meta_optimizer, epoch):
    model.train()
    correct = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        flat_data = data.view(-1, 28 * 28)
        if args.train_adv > 0.:
            flat_data.requires_grad = True
        output, loss = train_once(args, model, flat_data, target, meta_optimizer)
        if args.train_adv > 0.:
            # Creates an adversarial example to use for training.
            s = torch.sign(flat_data.grad)
            x_adversarial = torch.clamp(flat_data + torch.mul(s, args.train_adv), 0., 1.)
            train_once(args, model, x_adversarial, target, meta_optimizer)
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
            if args.rbf or args.sigmoid:
                confidence = output.max(1)[0] / torch.sum(output, 1)
            else:
                # The output is already a softmax.
                confidence = torch.max(torch.exp(output), 1)[0]
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

def test_adversarial(args, model, device, test_loader):
    epsilons = np.arange(0.0, 0.45, 0.05)
    correctnesses = []
    for epsilon in epsilons:
        correct = 0.
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            flat_data = data.view(-1, 28 * 28)
            flat_data.requires_grad = True # We want the gradient wrt the input.
            output = model(flat_data)
            loss = square_distance_loss(output, target) if args.rbf else (
                loss_sigmoid(output, target) if args.sigmoid else loss_relu(output, target))
            loss.backward()
            # Builds an adversarial input.
            s = torch.sign(flat_data.grad)
            x_adversarial = torch.clamp(flat_data + torch.mul(s, epsilon), 0., 1.)
            # And feeds it, measuring the correctness.
            output_adversarial = model(x_adversarial)
            pred_adversarial = output_adversarial.max(1, keepdim=True)[1]
            correct += pred_adversarial.eq(target.view_as(pred_adversarial)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    result = zip(epsilons, correctnesses)
    print("Performance under adversarial:")
    for epsilon, c in result:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result

def test_adversarial_multistep(args, model, device, test_loader):
    epsilons = np.arange(0.05, 0.45, 0.05)
    correctnesses = []
    for epsilon in epsilons:
        correct = 0.
        step_epsilon = epsilon / float(args.adversarial_steps)
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            x = data.view(-1, 28 * 28)
            for adversarial_step in range(args.adversarial_steps):
                x_input = torch.tensor(x)
                x_input.requires_grad = True
                output = model(x_input)
                loss = square_distance_loss(output, target) if args.rbf else (
                    loss_sigmoid(output, target) if args.sigmoid else loss_relu(output, target))
                loss.backward()
                # Builds the next adversarial input.
                # The step length is bounded by step_epsilon.
                # We need to do the computation in this funny way to leave the tensor on CUDA.
                delta = torch.mul(torch.reciprocal(torch.max(torch.abs(x_input.grad), -1, keepdim=True)[0]), step_epsilon)
                x = torch.clamp(x + delta * x_input.grad, 0., 1.)
            # Measures the correctness.
            output_adversarial = model(x)
            pred_adversarial = output_adversarial.max(1, keepdim=True)[1]
            correct += pred_adversarial.eq(target.view_as(pred_adversarial)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    result = zip(epsilons, correctnesses)
    print("Performance under multi adversarial:")
    for epsilon, c in result:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result

def test_adversarial_multistep_modinf(args, model, device, test_loader):
    epsilons = np.arange(0.05, 0.45, 0.05)
    correctnesses = []
    for epsilon in epsilons:
        correct = 0.
        step_epsilon = epsilon / float(args.adversarial_steps)
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            x = data.view(-1, 28 * 28)
            for adversarial_step in range(args.adversarial_steps):
                x_input = torch.tensor(x)
                x_input.requires_grad = True
                output = model(x_input)
                loss = square_distance_loss(output, target) if args.rbf else (
                    loss_sigmoid(output, target) if args.sigmoid else loss_relu(output, target))
                loss.backward()
                # Builds the next adversarial input.
                # The step length is bounded by step_epsilon.
                s = torch.sign(x_input.grad)
                x = torch.clamp(x + torch.mul(s, step_epsilon), 0., 1.)
            # Measures the correctness.
            output_adversarial = model(x)
            pred_adversarial = output_adversarial.max(1, keepdim=True)[1]
            correct += pred_adversarial.eq(target.view_as(pred_adversarial)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    result = zip(epsilons, correctnesses)
    print("Performance under multi adversarial modinf:")
    for epsilon, c in result:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result

def test_perturbation(args, model, device, test_loader):
    epsilons = np.arange(0.0, 0.55, 0.05)
    correctnesses = []
    for epsilon in epsilons:
        correct = 0.
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            flat_data = data.view(-1, 28 * 28)
            dims = list(flat_data.shape)
            s = flat_data.new(*dims)
            s.uniform_(0., 1.) # Uniform noise
            # Convex combination.
            x_perturbed = torch.clamp(torch.mul(flat_data, 1. - epsilon) + torch.mul(s, epsilon), 0., 1.)
            # And feeds it, measuring the correctness.
            output_perturbed = model(x_perturbed)
            pred_perturbed = output_perturbed.max(1, keepdim=True)[1]
            correct += pred_perturbed.eq(target.view_as(pred_perturbed)).sum().item()
        correctnesses.append(100. * correct / float(len(test_loader.dataset)))
    result = zip(epsilons, correctnesses)
    print("Performance under perturbations:")
    for epsilon, c in result:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result

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
    parser.add_argument('--rbf', action='store_true', default=False,
                        help='Use RBF nets')
    parser.add_argument('--andor', type=str, default='**v',
                        help='Type of neurons in RBF nets, ^ = and, v = or, * = random mix')
    parser.add_argument('--modinf', action='store_true', default=False,
                        help='Use infinity norm for RBFs')
    parser.add_argument('--regular_deriv', action='store_true', default=False,
                        help='Use regular derivative for training RBFs')
    parser.add_argument('--min_slope', type=float, default=0.01,
                        help='Minimum slope for RBF (default: 0.01)')
    parser.add_argument('--max_slope', type=float, default=3.0,
                        help='Maximum slope for RBF (default: 3.0)')
    parser.add_argument('--sensitivity_cost', type=float, default=0.,
                        help='Cost of sensitivity')
    parser.add_argument('--sigmoid', action='store_true', default=False,
                        help='Use sigmoid neurons (otherwise, relu)')
    parser.add_argument('--opt', type=str, default='ada',
                        help='"ada" for adadelta, "mom" for momentum')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 1 for lin and 5 for RBF)')
    parser.add_argument('--train_adv', type=float, default=0.0,
                        help='Train also on adversarial examples computed with given attack quantity')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--adv', action='store_true', default=False,
                        help='Estimate resistance to adversarial attacks')
    parser.add_argument('--adversarial_steps', type=int, default=4,
                        help='Number of steps in adversarial multistep testing')
    parser.add_argument('--tri', action='store_true', default=False,
                        help='Test random input')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs to perform')

    args = parser.parse_args()
    if args.lr is None:
        args.lr = 5. if args.rbf else 1.
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

    # Prepares dictionary of outputs.
    output = {'parameters': args.__dict__,
              'accuracy_test': [],
              'sensitivity': [],
              'accuracy_adv': [],
              'accuracy_adv_multi': [],
              'accuracy_adv_multi_modinf': [],
              'accuracy_pert': []}

    for run_idx in range(args.runs):
        print("====== Run {} ======".format(run_idx))
        print(json.dumps(args.__dict__))
        if args.runs > 1:
            torch.manual_seed(run_idx)
        # Creates the model.
        model = RBFNet(args).to(device) if args.rbf else StdNet(args).to(device)
        # Creates an optimizer.
        params = filter(lambda p: p.requires_grad, model.parameters())
        if args.opt == 'mom':
            optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
        else:
            optimizer = optim.Adadelta(params, lr=args.lr)
        # For RBF nets, we wrap it into an enforcer.
        meta_optimizer = ParamBoundEnforcer(optimizer) if args.rbf else optimizer
        # Trains the model for given number of epochs.
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, meta_optimizer, epoch)
        accuracy, sensitivity = test(args, model, device, test_loader)
        output['accuracy_test'].append(accuracy)
        output['sensitivity'].append(sensitivity)
        if args.adv:
            output['accuracy_adv'].append(test_adversarial(args, model, device, test_loader))
            output['accuracy_adv_multi'].append(test_adversarial_multistep(args, model, device, test_loader))
            output['accuracy_adv_multi_modinf'].append(test_adversarial_multistep_modinf(args, model, device, test_loader))
            output['accuracy_pert'].append(test_perturbation(args, model, device, test_loader))

    # Writes the results to file if requested.
    modifiers = []
    if args.rbf and args.modinf:
        modifiers.append('modinf')
    if args.rbf and args.regular_deriv:
        modifiers.append('regular_deriv')

    fn = "measure_{}_{}[{}]_{}runs_{}epochs{}.json".format(
        datetime.datetime.now().isoformat().replace(':', '-'),
        'rbf' if args.rbf else 'sigmoid' if args.sigmoid else 'relu',
        args.layers,
        args.runs,
        args.epochs,
        '_' + '_'.join(modifiers) if modifiers else ''
    )
    with open(fn, 'w') as f:
        json.dump(output, f,
                  sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()