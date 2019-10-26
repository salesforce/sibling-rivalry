# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from base.actors.base import BaseActor
from base.learners.distance import BaseDistanceLearner, BaseSiblingRivalryLearner
from agents.pixgrid_agents.modules import StochasticPolicy, Value
from agents.pixgrid_agents.pixgrid_env import Env


class StochasticAgent(BaseActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_keys = [
            'state', 'next_state', 'goal', 'mask',
            'action', 'n_ent', 'log_prob',
            'reward', 'terminal', 'complete'
        ]
        self.no_squeeze_list = [
            'state', 'next_state', 'goal'
        ]

    def _make_modules(self, policy):
        self.policy = policy

    def step(self, do_eval=False):
        s = self.env.state
        g = self.env.goal
        mask = self.env.action_mask()
        a, log_prob, n_ent = self.policy(s[None], g[None], mask[None], greedy=do_eval)
        a = a.view([])
        log_prob = log_prob.sum()
        n_ent = n_ent.mean()

        self.env.step(a)
        complete = float(self.env.is_success) * torch.ones(1)
        terminal = float(self.env.is_done) * torch.ones(1)
        s_next = self.env.state
        m_next = self.env.action_mask()
        r = -1 * torch.ones(1)

        self.episode.append({
            'state': s,
            'goal': g,
            'mask': mask,
            'action': a,
            'log_prob': log_prob,
            'n_ent': n_ent,
            'next_state': s_next,
            'next_mask': m_next,
            'reward': r.view([]),
            'terminal': terminal.view([]),
            'complete': complete.view([]),
        })

    @property
    def rollout(self):
        states = torch.stack([e['state'] for e in self.episode] + [self.episode[-1]['next_state']]).data.numpy()
        grids = states[:, 0]
        locs = states[:, 1]
        return grids, locs


class DistanceLearner(BaseDistanceLearner):
    def create_env(self):
        return Env(**self.env_params)

    def _make_agent_modules(self):
        self.policy = StochasticPolicy(self._dummy_env.W)
        self.v_module = Value(self._dummy_env.W, use_antigoal=False)

    def _make_agent(self):
        return StochasticAgent(env=self.create_env(), policy=self.policy)

    def get_values(self, batch):
        return self.v_module(
            batch['state'],
            batch['goal'],
            batch.get('antigoal', None)
        )

    def get_terminal_values(self, batch):
        if 'antigoal' in batch:
            antigoal = batch['antigoal'][-1:]
        else:
            antigoal = None
        return self.v_module(
            batch['next_state'][-1:],
            batch['goal'][-1:],
            antigoal
        )

    def get_policy_lprobs_and_nents(self, batch):
        _, log_prob, n_ent = self.policy(
            batch['state'], batch['goal'], batch['mask'],
            action=batch['action']
        )
        return log_prob, n_ent

    def get_icm_loss(self, batch):
        return self.icm(batch['state'], batch['next_state'], batch['action'])


class SiblingRivalryLearner(BaseSiblingRivalryLearner, DistanceLearner):
    def _make_agent_modules(self):
        self.policy = StochasticPolicy(self._dummy_env.W)
        self.v_module = Value(self._dummy_env.W, use_antigoal=self.use_antigoal)

    def _make_agent(self):
        agent = super()._make_agent()
        agent.batch_keys.append('antigoal')
        agent.no_squeeze_list.append('antigoal')
        return agent
