# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from .base import BaseLearner
import torch
import numpy as np


class BaseDistanceLearner(BaseLearner):
    AGENT_TYPE = 'Distance'
    def __init__(self,
                 shaped_type='terminal',
                 **kwargs):

        self.shaped_type = str(shaped_type).lower()
        assert self.shaped_type in [
            'off',        # No shaped reward (just sparse)
            'dense',      # Shaped reward provided at every timestep
            'dense_diff', # Shaped reward provided at every timestep as change in reward potential
            'terminal',   # Shaped reward only provided at terminal timestep (affects gamma and bootstrap_from_early_terminal)
        ]

        super().__init__(**kwargs)

        if self.shaped_type == 'terminal':
            self.gamma = 1.0
            self.bootstrap_from_early_terminal = False

        self.distance = []

    @property
    def was_success(self):
        return bool(self.agent.env.is_success)

    @property
    def n_steps(self):
        return int(len(self.agent.episode))

    @property
    def dist_to_goal(self):
        return float(self.distance[-1])

    def _reset_ep_stats(self):
        super()._reset_ep_stats()
        self.distance = []

    def distance_reward(self, transition):
        # No shaped reward (just sparse)
        if self.shaped_type == 'off':
            return torch.zeros_like(transition['reward'])

        # Shaped reward provided at every timestep
        elif self.shaped_type == 'dense':
            return -self._dummy_env.dist(transition['next_state'], transition['goal'])

        # Shaped reward provided at every timestep as change in reward potential
        elif self.shaped_type == 'dense_diff':
            ddg = -self._dummy_env.dist(transition['next_state'], transition['goal'])
            ddg += self._dummy_env.dist(transition['state'], transition['goal'])
            return ddg

        # Shaped reward only provided at terminal timestep
        elif self.shaped_type == 'terminal':
            ddg = -self._dummy_env.dist(transition['next_state'], transition['goal'])
            ddg *= transition['terminal']
            return ddg

    def relabel_episode(self):
        self._compress_me = []
        for e in self.agent.episode:
            r = e['complete'] + self.distance_reward(e)
            e['reward'] *= 0
            e['reward'] += r

            self.distance.append(self._dummy_env.dist(e['next_state'], e['goal']).item())

        self._compress_me.append(self.agent.episode)
        self._add_im_reward()

    def fill_summary(self, *values):
        manual_summary = [float(self.was_success), float(self.dist_to_goal)]

        for v in values:
            manual_summary.append(v.item())

        self._ep_summary = manual_summary


class BaseSiblingRivalryLearner(BaseDistanceLearner):
    AGENT_TYPE = 'SiblingRivalry'
    def __init__(self,
                 use_antigoal=True,
                 rect=True,
                 sibling_epsilon=0.0,
                 **kwargs):

        self.use_antigoal = bool(use_antigoal)
        self.rect = bool(rect)
        self.sibling_epsilon = max(0.0, sibling_epsilon)

        super().__init__(**kwargs)

        self.agents = [self.agent, self._make_agent()]
        self.a0 = self.agents[0]
        self.a1 = self.agents[1]
        self.distance = [[], []]
        self.distance_ag = [[], []]
        self._best_succeeded = False


    ################ BOOKKEEPING ####################
    @property
    def was_success(self):
        return bool(self._best_succeeded)

    @property
    def avg_success(self):
        return float(np.mean([agent.env.is_success for agent in self.agents]))

    @property
    def avg_dist_to_goal(self):
        return float(np.mean([d[-1] for d in self.distance]))

    @property
    def avg_dist_to_antigoal(self):
        return float(np.mean([d[-1] for d in self.distance_ag]))

    def _reset_ep_stats(self):
        super()._reset_ep_stats()
        self.distance = [[], []]
        self.distance_ag = [[], []]
        self._best_succeeded = False


    def fill_summary(self, *values):
        succeeded = self.avg_success
        dist_to_g = self.avg_dist_to_goal
        dist_to_a = self.avg_dist_to_antigoal
        manual_summary = [succeeded, dist_to_g, dist_to_a]

        for v in values:
            manual_summary.append(v.item())

        self._ep_summary = manual_summary

    ################ CORE ALGORITHM ####################
    def play_episode(self, reset_dict=None, do_eval=False, **kwargs):
        self._reset_ep_stats()

        if reset_dict is None:
            reset_dict = {}

        for agent in self.agents:
            agent.play_episode(reset_dict, do_eval, **kwargs)
            reset_dict = agent.env.sibling_reset
        self.relabel_episode()

    def relabel_episode(self):
        # Add achieved states of sibling as antigoal
        achieved0 = self.a0.env.achieved
        achieved1 = self.a1.env.achieved
        for e in self.a0.episode:
            e['antigoal'] = achieved1.detach()
        for e in self.a1.episode:
            e['antigoal'] = achieved0.detach()

        # Determine which of the siblings should be included for training
        goal = self.a0.env.goal.detach()
        is_0_closer = self._dummy_env.dist(goal, achieved0) < self._dummy_env.dist(goal, achieved1)
        within_epsilon = self._dummy_env.dist(achieved0, achieved1) < self.sibling_epsilon

        if is_0_closer:
            include0 = bool(within_epsilon) or self.a0.env.is_success
            include1 = True
            self._best_succeeded = bool(self.a0.env.is_success)
        else:
            include0 = True
            include1 = bool(within_epsilon) or self.a1.env.is_success
            self._best_succeeded = bool(self.a1.env.is_success)

        # Relabel rewards and set aside included rollouts for downstream training
        ep_tuples = [
            (0, self.a0.episode, include0),
            (1, self.a1.episode, include1)
        ]
        self._compress_me = []
        for ai, ep, include in ep_tuples:
            for e in ep:
                r = e['complete'] + self.distance_reward(e)
                e['reward'] *= 0
                e['reward'] += r

                self.distance[ai].append(self._dummy_env.dist(e['next_state'], e['goal']).item())
                self.distance_ag[ai].append(self._dummy_env.dist(e['next_state'], e['antigoal']).item())

            if include:
                self._compress_me.append(ep)

        self._add_im_reward()

    def distance_reward(self, transition):
        # No shaped reward (just sparse)
        if self.shaped_type == 'off':
            return torch.zeros_like(transition['reward'])

        # Shaped reward provided at every timestep
        elif self.shaped_type == 'dense':
            drg = -self._dummy_env.dist(transition['next_state'], transition['goal'])
            if not self.use_antigoal:
                return drg

            dra = -self._dummy_env.dist(transition['next_state'], transition['antigoal'])
            return torch.clamp(drg - dra, -np.inf, 0 if self.rect else np.inf)

        # Shaped reward provided at every timestep as change in reward potential
        elif self.shaped_type == 'dense_diff':
            drg = -self._dummy_env.dist(transition['next_state'], transition['goal'])
            drg += self._dummy_env.dist(transition['state'], transition['goal'])
            if not self.use_antigoal:
                return drg

            dra = -self._dummy_env.dist(transition['next_state'], transition['antigoal'])
            dra += self._dummy_env.dist(transition['state'], transition['antigoal'])
            return drg - dra  # This ignores self.rect

        # Shaped reward only provided at terminal timestep
        elif self.shaped_type == 'terminal':
            drg = -self._dummy_env.dist(transition['next_state'], transition['goal'])
            if not self.use_antigoal:
                return transition['terminal'] * drg

            dra = -self._dummy_env.dist(transition['next_state'], transition['antigoal'])
            return transition['terminal'] * torch.clamp(drg - dra, -np.inf, 0 if self.rect else np.inf)
