# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from __future__ import division
import time
import numpy as np
from torch.nn.utils import clip_grad_norm
from numpy.random import rand


def ensure_shared_grads(model, shared_model, gpu=False):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None and not gpu:
            return
        elif param.grad is None:
            continue
        elif gpu:
            shared_param._grad = param.grad.cpu()
        else:
            shared_param._grad = param.grad


def n_optimizations(optim):
    for group in optim.param_groups:
        continue
    for p in group['params']:
        continue
    return int(optim.state[p]['step'].item())



def lazy_shared_batch(model, shared_model, shared_optim, batch_size, gpu=False, max_norm=None):
    n_opt_pre = n_optimizations(shared_optim)
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if param.grad is None:
            continue
        elif gpu:
            grad_to_add = param.grad.cpu()
        else:
            grad_to_add = param.grad

        grad_to_add = grad_to_add / float(batch_size)

        if shared_param._grad is None:
            shared_param._grad = grad_to_add
        else:
            shared_param._grad += grad_to_add

    shared_model.examples_processed += 1
    n_proc = int(shared_model.examples_processed.item())

    if n_proc > batch_size:
        if shared_model.examples_initiated.item() < batch_size:
            return
        print('PROBLEM! We have processed too many examples: {}'.format(n_proc), flush=True)
        time.sleep(float(5 * rand()))
        n_opt_post = n_optimizations(shared_optim)

        # There was a race. You won. Complete the update.
        if n_opt_post == n_opt_pre:
            print('I won the race. Performing optimization #{}.'.format(n_opt_post), flush=True)
            if max_norm is not None:
                clip_grad_norm(shared_model.parameters(), max_norm=max_norm)
            shared_optim.step()
            shared_optim.zero_grad()
            shared_model.examples_initiated *= 0.
            shared_model.examples_processed *= 0.
        else:
            return

    if n_proc == batch_size:
        if max_norm is not None:
            clip_grad_norm(shared_model.parameters(), max_norm=max_norm)
        shared_optim.step()
        shared_optim.zero_grad()
        shared_model.examples_initiated *= 0.
        shared_model.examples_processed *= 0.


def param_sanity_check(model, shared_model, p_idx=0, prefix=''):
    i = 0
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if i == p_idx:
            print('{}   {:15.11f},  {:15.11f}'.format(prefix, param.data.view(-1)[0], shared_param.data.view(-1)[0]), flush=True)
            break
        i += 1


def grad_sanity_check(model, shared_model, p_idx=0, prefix=''):
    i = 0
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if i == p_idx:
            print(
                '{}   {:15.11f},  {:15.11f}'.format(
                    prefix, param.grad.data.view(-1)[0], shared_param.grad.data.view(-1)[0]
                ),
                flush=True
            )
            break
        i += 1


class Error(Exception):
    """Base clas for NaN Error"""
    pass

class NaNError(Error):
    """Raised when NaN(s) is/are encountered"""
    pass


def nan_check(model):
    def rec_modules(m, p_dict=None, prefix=None):
        if p_dict is None:
            p_dict = {}
        if prefix is None:
            prefix = []
        for k, v in m._modules.items():
            if v._modules:
                p_dict = rec_modules(v, p_dict, prefix + [k])
            else:
                for n, p in v.named_parameters():
                    full_keys = prefix + [n]
                    p_dict[', '.join(full_keys)] = p.data.numpy()
        return p_dict

    pd = rec_modules(model)

    has_nans = []
    for k, v in pd.items():
        if np.isnan(v).any():
            has_nans.append(k)

    if has_nans:
        pstr = 'Found nans in the following parameters:'
        for k in has_nans:
            pstr += '\n\t{}'.format(k)
        print(pstr, flush=True)
        raise NaNError


def query_yes_no(question):
    """Ask a yes/no question and return their answer.

    "question" is a string that is presented to the user.

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}

    prompt = question + '  [y/n]  '
    print('')
    while True:
        choice = input(prompt)
        choice = choice.lower()
        if choice in valid:
            print('')
            return valid[choice]
        else:
            print('Response not valid. Please answer yes or no.', flush=True)


def interpret_duration(d_str):
    d_str = d_str.lower()

    assert d_str[-1] in ['s', 'm', 'h', 'd']
    assert all([c in '.0123456789' for c in d_str[:-1]])

    d = float(d_str[:-1])
    t = d_str[-1]

    if t == 's':
        return d
    if t == 'm':
        return d * 60
    if t == 'h':
        return d * 60 * 60
    if t == 'd':
        return d * 60 * 60 * 24


def date_string():
    lt = time.localtime()
    return '_'.join(
        [str(lt.tm_year)] + ['%02d' % d for d in [lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec]]
    )




