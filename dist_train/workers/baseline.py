# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import os
import json
import time
import torch
import numpy as np
from dist_train.utils.shared_optim import SharedAdam as Adam
from dist_train.workers.base import OffPolicyManager, OnPolicyManager, PPOManager


class OffPolicy(OffPolicyManager):

    def rollout_wrapper(self, c_ep_counter):
        st = time.time()
        self.agent_model.play_episode()
        # Add episode for training.
        self.replay_buffer.add_episode(self.agent_model.transitions_for_buffer(training=True))
        dur = time.time() - st

        # Calculate losses to allow dense logging
        episode_stats = self.agent_model.episode_summary()

        self._log_rollout(c_ep_counter, dur, episode_stats)

    def _log_rollout(self, c_ep_counter, dur, episode_stats):
        # Increment the steps counters, place the log in the epoch buffer, and give a quick rollout print
        c_ep_counter += 1
        self.time_keeper['n_rounds'] += 1

        n_steps = int(self.agent_model.train_steps.data.item()) + int(c_ep_counter.item())
        timestamp = ''.join('{:017.4f}'.format(time.time()).split('.'))
        log = {'{:d}.{}'.format(n_steps, timestamp): [str(sl) for sl in episode_stats]}
        self.epoch_buffer.append(log)

        dense_save = False  # (int(self.time_keeper['n_rounds']) % self.settings.ep_save) == 0 and self.rank == 0

        log_str = '{:10d} - {} {:6d}   Dur = {:6.2f}, Steps = {:3d} {} {}'.format(
            n_steps,
            '*' if dense_save else ' ',
            int(self.time_keeper['n_rounds']),
            dur,
            int(self.agent_model.n_steps),
            '!!!' if int(self.agent_model.was_success) else '   ',
            '*' if dense_save else ' '
        )
        self.logger.info(log_str)

    def eval_wrapper(self):
        stats = []
        episodes = {}
        for evi in range(self.config.get('eval_iters', 10)):
            self.agent_model.play_episode(do_eval=self.config.get('greedy_eval', True))

            ep_stats = [float(x) for x in self.agent_model.episode_summary()]
            stats.append(ep_stats)

            dump_ep = []
            for t in self.agent_model.curr_ep:
                dump_t = {k: np.array(v.detach()).tolist() for k, v in t.items()}
                dump_ep.append(dump_t)
            episodes[evi] = dump_ep

        return stats, episodes


class HierarchicalOffPolicy(OffPolicy):
    def __init__(self, rank, config, settings):
        super().__init__(rank, config, settings)

        self.optim_lo_path = os.path.join(self.exp_dir, 'optim_lo.pth.tar')

        self.optim_lo = Adam(self.agent_model._lo_parameters, lr=config['learning_rate'])
        if os.path.isfile(self.optim_lo_path):
            self.optim_lo.load_state_dict(torch.load(self.optim_lo_path))

    def checkpoint(self):
        super().checkpoint()
        torch.save(self.optim_lo, self.optim_lo_path)

    def rollout_wrapper(self, c_ep_counter):
        st = time.time()
        self.agent_model.play_episode(optim_lo=self.optim_lo)
        self.agent_model.relabel_episode()
        # Add episode for training.
        self.replay_buffer.add_episode(self.agent_model.transitions_for_buffer(training=True))
        dur = time.time() - st

        # Calculate losses to allow dense logging
        episode_stats = self.agent_model.episode_summary()

        self._log_rollout(c_ep_counter, dur, episode_stats)


class OnPolicy(OnPolicyManager):
    def rollout_wrapper(self, c_ep_counter):
        st = time.time()
        self.agent_model.eval()
        self.agent_model.play_episode()

        self.agent_model.train()
        loss = self.condense_loss(self.agent_model())
        dur = time.time() - st

        # Calculate losses to allow dense logging
        episode_stats = self.agent_model.episode_summary()

        self._log_rollout(c_ep_counter, dur, episode_stats)

        return loss

    def _log_rollout(self, c_ep_counter, dur, episode_stats):
        c_ep_counter += 1
        n_steps = int(self.agent_model.train_steps.data.item()) + int(c_ep_counter.item())
        timestamp = ''.join('{:017.4f}'.format(time.time()).split('.'))

        dense_save = False  # (int(self.time_keeper['n_rounds']) % self.settings.ep_save) == 0 and self.rank == 0

        # The burden to save falls to us
        if dense_save:
            dstr = '{:010d}.{}'.format(n_steps, timestamp)
            config_path = self.settings.config_path
            exp_name = config_path.split('/')[-1][:-5]
            exp_dir = os.path.join(self.settings.log_dir, exp_name)
            c_path = os.path.join(exp_dir, dstr + '.json')
            dump_ep = []
            for t in self.agent_model.curr_ep:
                dump_t = {k: np.array(v.detach()).tolist() for k, v in t.items()}
                dump_ep.append(dump_t)
            with open(c_path, 'wt') as f:
                json.dump(dump_ep, f)
            self.time_keeper['ep_save'] = int(self.time_keeper['n_rounds'])

        # Increment the steps counters and log the results.
        self.time_keeper['n_rounds'] += 1
        hist_name = 'hist_{}.json'.format(self.rank)
        with open(os.path.join(self.exp_dir, hist_name), 'a') as save_file:
            log = {'{:d}.{}'.format(n_steps, timestamp): [str(sl) for sl in episode_stats]}
            save_file.write(json.dumps(log))
            save_file.close()

        log_str = '{:10d} - {} {:6d}   Dur = {:6.2f}, Steps = {:3d} {} {}'.format(
            n_steps,
            '*' if dense_save else ' ',
            int(self.time_keeper['n_rounds']),
            dur,
            int(self.agent_model.n_steps),
            '!!!' if int(self.agent_model.was_success) else '   ',
            '*' if dense_save else ' '
        )
        self.logger.info(log_str)

    def eval_wrapper(self):
        stats = []
        episodes = {}
        for evi in range(self.config.get('eval_iters', 10)):
            self.agent_model.play_episode(do_eval=bool(self.config.get('greedy_eval', True)))

            ep_stats = [float(x) for x in self.agent_model.episode_summary()]
            stats.append(ep_stats)

            dump_ep = []
            for t in self.agent_model.curr_ep:
                dump_t = {k: np.array(v.detach()).tolist() for k, v in t.items()}
                dump_ep.append(dump_t)
            episodes[evi] = dump_ep

        return stats, episodes


class PPO(PPOManager, OnPolicy):
    def rollout_wrapper(self, c_ep_counter):
        st = time.time()
        self.agent_model.reach_horizon()
        dur = time.time() - st

        # Calculate losses to allow dense logging
        episode_stats = self.agent_model.episode_summary()

        self._log_rollout(c_ep_counter, dur, episode_stats)


class HierarchicalPPO(PPO):
    def __init__(self, rank, config, settings):
        super().__init__(rank, config, settings)

        self.optim_lo_path = os.path.join(self.exp_dir, 'optim_lo.pth.tar')

        self.optim_lo = Adam(self.agent_model._lo_parameters, lr=config['learning_rate'])
        if os.path.isfile(self.optim_lo_path):
            self.optim_lo.load_state_dict(torch.load(self.optim_lo_path))

    def checkpoint(self):
        super().checkpoint()
        torch.save(self.optim_lo, self.optim_lo_path)

    def rollout_wrapper(self, c_ep_counter):
        st = time.time()
        self.agent_model.reach_horizon(optim_lo=self.optim_lo)
        dur = time.time() - st

        # Calculate losses to allow dense logging
        episode_stats = self.agent_model.episode_summary()

        self._log_rollout(c_ep_counter, dur, episode_stats)