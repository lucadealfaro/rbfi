#!/usr/bin/env python

# Author: Luca de Alfaro, 2018
# License: BSD

# Permutation-invariant MNIST.

from __future__ import print_function
import argparse
import datetime
import json
from json_plus import Serializable, Storage

import torch
from torch import nn
from nn_linear_pytorch_sensitivity import LinearWithSensitivity
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
from rbfi import RBFI
from square_distance_loss import square_distance_loss
from torch_bounded_parameters import ParamBoundEnforcer
from test_pgd import pgd_attack, pgd_batch_attack, compute_pgd_example

# Loss functions
loss_relu = F.nll_loss
loss_sigmoid = square_distance_loss

cross_entropy_loss = nn.CrossEntropyLoss()
nll_loss = nn.NLLLoss()

def loss_soft_dist(output, target):
    soft_output = F.softmax(output, dim=1)
    return square_distance_loss(soft_output, target)

def get_loss_function(args):
    if args.rbf:
        return square_distance_loss
    elif args.sigmoid:
        return loss_sigmoid
    else:
        return loss_relu

class StdNet(nn.Module):
    """Standard neural net, which can be sigmoid or RELU, depending on flags."""
    def __init__(self, args, empty=False):
        super(StdNet, self).__init__()
        self.args = args
        layer_sizes = map(int, args.layers.split(',')) + [10] # 10 for the output.
        self.layers = []
        if not empty:
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

    def regularization(self):
        """This function is purely for homogeneity with RBFNet."""
        return torch.tensor(0)

    def set_regular_deriv(self, b):
        """This function is purely for homogeneity with RBFNet."""
        pass

    # Serialization.
    def dumps(self):
        d = dict(
            args=self.args.__dict__,
            layers = [l.dumps() for l in self.layers]
        )
        return Serializable.dumps(d)

    @staticmethod
    def loads(s, device):
        d = Serializable.loads(s)
        m = StdNet(d['args'], empty=True)
        for i, ms in enumerate(d['layers']):
            l = LinearWithSensitivity.loads(ms, device)
            m.layers.append(l)
            m.add_module('layer_%d' % i, l)
        return m


class RBFNet(nn.Module):
    """RBF Neural net."""

    def __init__(self, args, empty=False):
        super(RBFNet, self).__init__()
        self.args = args
        layer_sizes = map(int, args.layers.split(',')) + [10] # 10 for the output.
        self.layers = []
        if not empty:
            previous_size = 28 * 28
            for i, size in enumerate(layer_sizes):
                l = RBFI(previous_size, size,
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

    def regularization(self):
        s = torch.mean(torch.abs(self.layers[0].u))
        for l in self.layers[2:]:
            s = s + torch.mean(torch.abs(l.u))
        return s

    def set_regular_deriv(self, b):
        for l in self.layers:
            l.regular_deriv=b

    def dumps(self):
        d = dict(
            args=self.args.__dict__,
            layers = [l.dumps() for l in self.layers]
        )
        return Serializable.dumps(d)

    @staticmethod
    def loads(s, device):
        d = Serializable.loads(s)
        args = Storage(d['args'])
        m = RBFNet(args, empty=True)
        for i, ms in enumerate(d['layers']):
            l = RBFI.loads(ms, device)
            m.layers.append(l)
            m.add_module('layer_%d' % i, l)
        return m


def read_model(in_fn, device):
    with open(in_fn, "r") as f:
        d = json.load(f)
        if d['kind'] == 'StdNet':
            model = StdNet.loads(d['model'], device)
        else:
            model = RBFNet.loads(d['model'], device)
    return model


def write_model(m, out_fn):
    d = dict(
        kind = 'StdNet' if isinstance(m, StdNet) else 'RBFNet',
        model = m.dumps()
    )
    with open(out_fn, 'w') as f:
        json.dump(d, f)


def train_once(args, loss_function, model, flat_data, target, meta_optimizer):
    if args.rbf:
        # For RBF, the optimizer also enforces parameter bounds.
        meta_optimizer.optimizer.zero_grad()
        output = model(flat_data)
        primary_loss = loss_function(output, target)
        loss = (primary_loss +
                + args.sensitivity_cost * model.sensitivity()
                + args.l1_regularization * model.regularization()
                )
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
    loss_function = get_loss_function(args)
    for batch_idx, (data_cpu, target_cpu) in enumerate(train_loader):
        data_cpu = data_cpu.view(-1, 28 * 28)
        x_input, target = data_cpu.to(device), target_cpu.to(device)
        if args.train_adv > 0.:
            x_input.requires_grad = True
        output, loss = train_once(args, loss_function, model, x_input, target, meta_optimizer)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        # Adversarial training, if requested.
        if args.train_adv > 0.:
            if args.use_pgd_for_adv_training:
                x_input_adv = compute_pgd_example(model, device, data_cpu, target_cpu, args.train_adv,
                                                  loss_function, num_iterations=args.train_adv_steps)
                # We check that the input is in the proper distance ball.
                assert torch.max(torch.abs(x_input - x_input_adv)).item() < args.train_adv * 1.01
                train_once(args, loss_function, model, x_input_adv, target, meta_optimizer)
            else:
                step_epsilon = args.train_adv / float(args.train_adv_steps)
                for adversarial_step in range(args.train_adv_steps):
                    # Creates an adversarial example to use for training.
                    s = torch.sign(x_input.grad)
                    new_x_input = torch.clamp(x_input + torch.mul(s, step_epsilon), 0., 1.)
                    x_input = x_input.new(*(new_x_input.shape))
                    x_input.requires_grad = True
                    x_input.data = new_x_input.data
                    output = model(x_input)
                    if adversarial_step < args.train_adv_steps - 1:
                        loss = loss_function(output, target)
                        loss.backward()
                    else:
                        train_once(args, loss_function, model, x_input, target, meta_optimizer)
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tAccuracy:{:.5f} Sensitivity: {}'.format(
                epoch, (batch_idx + 1) * len(data_cpu), len(train_loader.dataset),
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
            _, prediction = output.max(1, keepdim=True)
            if args.rbf or args.sigmoid:
                confidence = output.max(1)[0] / torch.sum(output, 1)
            else:
                # The output is already a softmax.
                confidence = torch.max(torch.exp(output), 1)[0]
            correct += prediction.eq(target.view_as(prediction)).sum().item()
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

def test_fsgm(args, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    correctnesses = []
    if not args.pseudo_adv:
        model.set_regular_deriv(True)
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
    model.set_regular_deriv(args.regular_deriv)
    result = zip(epsilons, correctnesses)
    print("Performance under adversarial:")
    for epsilon, c in result:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result

def test_adversarial_multistep(args, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    correctnesses = []
    if not args.pseudo_adv:
        model.set_regular_deriv(True)
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
    model.set_regular_deriv(args.regular_deriv)
    result = zip(epsilons, correctnesses)
    print("Performance under multi adversarial:")
    for epsilon, c in result:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result

def test_ifgsm(args, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    correctnesses = []
    if not args.pseudo_adv:
        model.set_regular_deriv(True)
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
    model.set_regular_deriv(args.regular_deriv)
    result = zip(epsilons, correctnesses)
    print("Performance under multi adversarial modinf:")
    for epsilon, c in result:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, c))
    return result

def test_perturbation(args, model, device, test_loader):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
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


def test_pgd(args, model, device, kwargs):
    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.01, 0.05)
    if not args.pseudo_adv:
        model.set_regular_deriv(True)
    pgd_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.pgd_batch, shuffle=True, **kwargs)
    loss_function = square_distance_loss if args.rbf else (
        loss_sigmoid if args.sigmoid else loss_relu)
    success_rate = [] # Success rate of the model, not of the attack, to be homogeneous.
    accuracy_vs_restarts_by_epsilon = {}
    for epsilon in epsilons:
        if args.pgd_batch == 1:
            # We do one by one, stopping as soon as we get success.
            attempted_attacks, successful_attacks = 0., 0.
            for i, (data, target) in enumerate(pgd_loader):
                attempted_attacks += 1.
                flat_data = data.view(-1, 28 * 28)
                if pgd_attack(model, device, flat_data, target, epsilon, loss_function,
                            num_iterations=args.pgd_iterations,
                            num_restarts=args.pgd_restarts):
                    successful_attacks += 1.
                if i == args.pgd_inputs:
                    break
            success_rate.append(successful_attacks / attempted_attacks)
        else:
            # We do in batches.
            fractions = []
            # I need to average the accuracy vs restarts on all batches.
            # So I accumulate the results in a list of numpy vectors.
            accuracy_vs_restart_list = []
            for i, (data, target) in enumerate(pgd_loader):
                flat_data = data.view(-1, 28 * 28)
                avs, fraction = pgd_batch_attack(
                    model, device, flat_data, target, epsilon, loss_function,
                    num_iterations=args.pgd_iterations,
                    num_restarts=args.pgd_restarts)
                fractions.append(fraction)
                accuracy_vs_restart_list.append(np.array(avs))
                if i + 1 >= args.pgd_inputs / args.pgd_batch:
                    break
            success_rate.append(float(np.mean(fractions)))
            # Now I need to average the success rate by epsilon.
            accuracy_vs_restart = list(np.average(np.vstack(accuracy_vs_restart_list), 0))
            accuracy_vs_restart = [float(x) for x in accuracy_vs_restart]
            accuracy_vs_restarts_by_epsilon[epsilon] = accuracy_vs_restart
    model.set_regular_deriv(args.regular_deriv)
    accuracy = zip(epsilons, success_rate)
    print("Performance under PGD:")
    for epsilon, c in accuracy:
        print("  Epsilon: {:.2f} Correctness: {}".format(epsilon, 1. - c))
    return accuracy, accuracy_vs_restarts_by_epsilon


def flatten(s):
    out = []
    for t in s:
        if isinstance(t, str):
            out.append(t)
        else:
            out += t
    return out


def split_arg_list(s):
    return flatten([t.split("=") for t in s.split()])


def main(arg_string=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--epochs_before_regular_deriv', type=int, default=0,
                        help='number of epochs to train with pseudoderivatives')
    parser.add_argument('--layers', type=str, default="32,32,32",
                        help='comma-separated list of layer sizes')
    parser.add_argument('--rbfi', action='store_true', default=False,
                        help='Use RBFI nets')
    parser.add_argument('--rbf', action='store_true', default=False,
                        help='Use RBF nets')
    parser.add_argument('--relu', action='store_true', default=False,
                        help='Use ReLU neurons')
    parser.add_argument('--andor', type=str, default='^v^v',
                        help='Type of neurons in RBF nets, ^ = and, v = or, * = random mix')
    parser.add_argument('--modinf', action='store_true', default=False,
                        help='Use infinity norm for RBFs')
    parser.add_argument('--regular_deriv', action='store_true', default=False,
                        help='Use true derivative for training RBFs')
    parser.add_argument('--min_slope', type=float, default=0.01,
                        help='Minimum slope for RBF (default: 0.01)')
    parser.add_argument('--max_slope', type=float, default=3.0,
                        help='Maximum slope for RBF (default: 3.0)')
    parser.add_argument('--sensitivity_cost', type=float, default=0.,
                        help='Cost of sensitivity')
    parser.add_argument('--l1_regularization', type=float, default=0.,
                        help='L1 regularization for RBF only')
    parser.add_argument('--init_slope', type=float, default=0.25,
                        help='Coefficient for slope initialization')
    parser.add_argument('--sigmoid', action='store_true', default=False,
                        help='Use sigmoid neurons (otherwise, relu)')
    parser.add_argument('--opt', type=str, default='ada',
                        help='"ada" for adadelta, "mom" for momentum')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: 1 for lin and 5 for RBF)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--train_adv', type=float, default=0.0,
                        help='Train also on adversarial examples computed with given attack quantity')
    parser.add_argument('--train_adv_steps', type=int, default=1,
                        help='Number of steps ')
    parser.add_argument('--pseudo_adv', action='store_true', default=False,
                        help='Use pseudoderivative in adversarial attacks.')
    parser.add_argument('--use_pgd_for_adv_training', action='store_true', default=False,
                        help='Use PGD for generating adversarial examples during training.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--test_fgsm', action='store_true', default=False,
                        help='Perform FSGM adversarial test.')
    parser.add_argument('--test_ifgsm', action='store_true', default=False,
                        help='Perform I-FSGM adversarial test.')
    parser.add_argument('--test_pert', action='store_true', default=False,
                        help='Perform random perturbation testing.')
    parser.add_argument('--test_pgd', action='store_true', default=False,
                        help='Perform PGD adversarial testing.')
    parser.add_argument('--pgd_iterations', type=int, default=100,
                        help="Number of PGD iterations"),
    parser.add_argument('--pgd_restarts', type=int, default=10,
                        help="Number of PGD restarts"),
    parser.add_argument('--pgd_inputs', type=int, default=1000,
                        help="Number of inputs to attack via PGD")
    parser.add_argument('--pgd_batch', type=int, default=100,
                        help="PGD batch size")
    parser.add_argument('--epsilon_min', type=float, default=0.05,
                        help="Min epsilon for testing attacks (must be multiple of 0.05)")
    parser.add_argument('--epsilon_max', type=float, default=0.5,
                        help="Max epsilon for testing attacks (must be multiple of 0.05)")
    parser.add_argument('--adversarial_steps', type=int, default=10,
                        help='Number of steps in adversarial multistep testing')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs to perform')
    parser.add_argument('--model_file', type=str, default=None,
                        help='Model file from which to read the models.')
    parser.add_argument('--continue_training', action='store_true', default=False,
                        help='Continue training a given model')
    parser.add_argument('--try_new_loss', action='store_true', default=False,
                        help='Try new loss function for training RBFI.')

    if arg_string is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(split_arg_list(arg_string))
    if not (args.relu or args.sigmoid):
        args.rbfi = True
    # RBFI use rbf and modinf.
    if args.rbfi:
        args.rbf = True
        args.modinf = True
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
              'accuracy_pert': [],
              'accuracy_pgd': [],
              'accuracy_vs_pgd_restarts': [],
              }

    if args.model_file is not None:
        out_fn = "measure_{}_[[{}]]".format(
            datetime.datetime.now().isoformat().replace(':', '-'),
            args.model_file)
    else:
        modifiers = []
        if args.rbf and args.modinf:
            modifiers.append('modinf')
            modifiers.append('slope{}'.format(args.max_slope))
            modifiers.append('senscost{}'.format(args.sensitivity_cost))
        if args.rbf and args.regular_deriv:
            modifiers.append('regular_deriv')
        out_fn = "measure_{}_{}[{}]_{}runs_{}epochs{}".format(
            datetime.datetime.now().isoformat().replace(':', '-'),
            'rbf' if args.rbf else 'sigmoid' if args.sigmoid else 'relu',
            args.layers,
            args.runs,
            args.epochs,
            '_' + '_'.join(modifiers) if modifiers else ''
        )

    for run_idx in range(args.runs):
        print("====== Run {} ======".format(run_idx))
        print(json.dumps(args.__dict__))
        if args.runs > 1:
            torch.manual_seed(run_idx)
        # Creates the model.
        if args.model_file is None:
            # We create a new model.
            model = RBFNet(args).to(device) if args.rbf else StdNet(args).to(device)
        else:
            # We read a model.
            if args.runs > 1:
                in_fn = args.model_file + '_{}.model.json'.format(run_idx)
            else:
                in_fn = args.model_file + '.model.json'
            model = read_model(in_fn, device)
            # Takes care of a few flags.
            model.set_regular_deriv(args.regular_deriv)
        if args.model_file is None or args.continue_training:
            # Trains the model.
            # Creates an optimizer.
            params = filter(lambda p: p.requires_grad, model.parameters())
            if args.opt == 'mom':
                optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum)
            elif args.opt == 'adam':
                optimizer = optim.Adam(params, lr=args.lr)
            else:
                optimizer = optim.Adadelta(params, lr=args.lr)
            # For RBF nets, we wrap it into an enforcer.
            meta_optimizer = ParamBoundEnforcer(optimizer) if args.rbf else optimizer
            # Trains the model for given number of epochs.
            for epoch in range(1, args.epochs + 1):
                if args.epochs_before_regular_deriv > 0 and epoch > args.epochs_before_regular_deriv:
                    model.set_regular_deriv(True)
                train(args, model, device, train_loader, meta_optimizer, epoch)
            # We write the model.
            model_fn = out_fn + '_{}.model.json'.format(run_idx)
            write_model(model, model_fn)
        # Now, we do the tests.
        accuracy, sensitivity = test(args, model, device, test_loader)
        output['accuracy_test'].append(accuracy)
        output['sensitivity'].append(sensitivity)
        if args.test_fgsm:
            output['accuracy_adv'].append(test_fsgm(args, model, device, test_loader))
        if args.test_ifgsm:
            output['accuracy_adv_multi_modinf'].append(test_ifgsm(args, model, device, test_loader))
        if args.test_pert:
            output['accuracy_pert'].append(test_perturbation(args, model, device, test_loader))
        if args.test_pgd:
            accuracy_pgd, accuracy_vs_restarts_by_epsilon = test_pgd(args, model, device, kwargs)
            output['accuracy_pgd'].append(accuracy_pgd)
            output['accuracy_vs_pgd_restarts'].append(accuracy_vs_restarts_by_epsilon)

    with open(out_fn + '.json', 'w') as f:
        json.dump(output, f,
                  sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    main()