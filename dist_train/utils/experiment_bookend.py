# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
from .helpers import nan_check
from .shared_optim import SharedAdam
from agents import agent_classes
import argparse
import time
import shutil
import json
import torch


def open_experiment(apply_time_machine=True):

    parser = argparse.ArgumentParser("Perform distributed RL.")

    # Required arguments
    parser.add_argument('--config-path', type=str,
                        help='Path to experiment config file (expecting a json)')

    parser.add_argument('--log-dir', type=str,
                        help='Parent directory that holds experiment log directories')

    parser.add_argument('--dur', type=str, help='Num epochs')

    parser.add_argument('--N', type=int, help='Number of workers')

    # Optional arguments

    parser.add_argument('--profile', type=str, default='15m',
                        help='Time between trainer profiler printouts (default = "15m" [15 minutes])')

    parser.add_argument('--keep_checkpoints', action='store_true',
                        help='Flag to enable saving model parameters separately for each checkpoint')

    args = parser.parse_args()

    config_path = args.config_path
    assert os.path.isfile(config_path)
    config = json.load(open(config_path))

    exp_name = config_path.split('/')[-1][:-5]
    exp_dir = os.path.join(args.log_dir, exp_name)

    print('Experiment directory is: {}'.format(exp_dir), flush=True)
    model_path = os.path.join(exp_dir, 'model.pth.tar')
    optim_path = os.path.join(exp_dir, 'optim.pth.tar')

    AgentClass = agent_classes(config['agent_type'], config['learner_type'], config['train_type'])

    # Create and/or restore the model
    model = AgentClass(**config['agent_params'])

    def _prepare_exp_dir():
        os.makedirs(exp_dir)
        shutil.copyfile(config_path, os.path.join(exp_dir, 'config.json'))

    if not os.path.isdir(exp_dir):
        _prepare_exp_dir()

    if apply_time_machine:
        if len(os.listdir(exp_dir)) > 0:
            # if query_yes_no('Restarting will erase all data. Continue with restart?'):
            shutil.rmtree(exp_dir)
            _prepare_exp_dir()

    if os.path.isfile(model_path):
        print('\nResuming where we left off!\n', flush=True)
        model.load_checkpoint(model_path)

    model.share_memory()

    shared_optimizer = SharedAdam(model.parameters(), lr=config['learning_rate'])
    if os.path.isfile(optim_path):
        shared_optimizer.load_state_dict(torch.load(optim_path))
    shared_optimizer.share_memory()
    shared_optimizer.zero_grad()

    return model, shared_optimizer, config, args


def close_experiment(model, optimizer, args):
    config_path = args.config_path
    exp_name = config_path.split('/')[-1][:-5]
    exp_dir = os.path.join(args.log_dir, exp_name)

    model_path = os.path.join(exp_dir, 'model.pth.tar')
    optim_path = os.path.join(exp_dir, 'optim.pth.tar')

    nan_check(model)
    if os.path.isfile(model_path):
        n = int(model.train_steps.item())
        model.save_checkpoint(os.path.join(exp_dir, '{:010d}_model.pth.tar'.format(n)))
        torch.save(optimizer.state_dict(), os.path.join(exp_dir, '{:010d}_optim.pth.tar'.format(n)))

        model.save_checkpoint(model_path)
        torch.save(optimizer.state_dict(), optim_path)

    n_episodes_played = int(model.train_steps.data.item())

    optimization_steps = 0
    tmp = optimizer
    for pg in tmp.param_groups:
        for p in pg['params']:
            optimization_steps = max(optimization_steps, int(tmp.state[p]['step'][0]))

    print(
        '\nDone!  --  N Episodes = {}  --  N Optimizations = {}\n'.format(
            n_episodes_played, optimization_steps
        ),
        flush=True
    )

    lt = time.localtime()
    dstr = '_'.join(
        [str(lt.tm_year)] + ['%02d' % d for d in [lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec]]
    )
    print(dstr, flush=True)