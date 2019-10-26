# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import time
import torch
import logging
import numpy as np

def create_worker_logger(worker_id, exp_dir, group_id=None):
    # create logger
    logger = logging.getLogger('{:02d}'.format(worker_id))
    logger.setLevel(logging.DEBUG)

    # # create file formatter
    # log_name = '{}.log'.format(worker_id) if group_id is None else '{}.{}.log'.format(worker_id, group_id)
    # fh = logging.FileHandler(filename=os.path.join(exp_dir, log_name))
    # fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(message)s'))
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    # create console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(name)s - %(message)s'))
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    return logger


class ReplayBuffer:
    def __init__(self, model, config):
        self.model = model

        self.capacity = config['buffer_capacity']
        self.batch_size = config['batch_size']
        self.min_size = max(config['min_buffer_size'], config['batch_size'])

        self.ep_buffer = [{}] * self.capacity
        self.opt_count = [0] * self.capacity

        self._pointer = 0
        self._looped = False

        self.profiler = {
            'st': time.time(),
            'last_idle': time.time(),
            'time_idle': 0,
            'time_sync': 0,
            'time_batch': 0,
            'size': [],
            'opts': [],
        }

    def reset_profiler(self):
        self.profiler = {
            'st': time.time(),
            'last_idle': time.time(),
            'time_idle': 0,
            'time_sync': 0,
            'time_batch': 0,
            'size': [],
            'opts': [],
        }

    @property
    def size(self):
        return int(self.capacity) if self._looped else int(self._pointer)

    def add_episode(self, transition_dicts):
        """Integrate new episodes from the workers into the buffers"""
        self.profiler['time_idle'] += time.time() - self.profiler['last_idle']
        st = time.time()
        self.ep_buffer[self._pointer] = {}
        self.opt_count[self._pointer] = 0
        for t in transition_dicts:
            self.ep_buffer[self._pointer] = {k: v.detach() for k, v in t.items()}
            self._pointer += 1
            if self._pointer >= self.capacity:
                self._looped = True
                self._pointer = 0

        self.profiler['size'].append(self.size)
        self.profiler['time_sync'] += time.time() - st
        self.profiler['last_idle'] = time.time()

    def make_batch(self, normalize=True):
        """Convert the buffer into inputs for DataParallel training"""
        self.profiler['time_idle'] += time.time() - self.profiler['last_idle']
        st = time.time()
        keys = self.ep_buffer[0].keys()
        batch = dict()
        ep_idxs = np.random.permutation(np.arange(self.size))[:self.batch_size]
        for k in keys:
            batch[k] = torch.stack([self.ep_buffer[i][k] for i in ep_idxs]).detach()
        for i in ep_idxs:
            self.opt_count[i] += 1

        if normalize:
            # Apply normalization to the batch
            batch = self.model.normalize_batch(batch)

        # Profile time statistics
        self.profiler['time_batch'] += time.time() - st
        self.profiler['last_idle'] = time.time()

        return batch

    def profile(self, time_window=900.):
        time_window = float(time_window)
        dur = time.time() - self.profiler['st']
        if dur < time_window:
            return

        print('\nTrainer time expenditure:\n  Syncing: {:5.2f}%,  Batching: {:5.2f}%  Training: {:5.2f}%'.format(
            100*self.profiler['time_sync']/dur,
            100*self.profiler['time_batch']/dur,
            100*self.profiler['time_idle']/dur,
        ), flush=True)
        # print('Average buffer size: {:7.1f}'.format(np.mean(self.profiler['size'])), flush=True)
        # print('Average episode use: {:7.1f}'.format(np.mean(self.profiler['opts'])), flush=True)
        # print(' ', flush=True)
        print('Current buffer size: {:7.1f}\n'.format(self.size), flush=True)

        self.reset_profiler()