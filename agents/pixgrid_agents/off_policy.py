# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np
from base.actors.base import BaseActor
from base.learners.distance import BaseDistanceLearner, BaseSiblingRivalryLearner
from base.learners.her import BaseHERLearner
from agents.pixgrid_agents.modules import Value
from agents.pixgrid_agents.pixgrid_env import Env


class QAgent(BaseActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.epsilon is not None

    def _make_modules(self, q_module):
        self.q_module = q_module

    def step(self, do_eval=False):
        s = self.env.state
        g = self.env.goal
        mask = self.env.action_mask()
        qs = self.q_module(s[None], g[None])[0]
        qs = qs.masked_fill(mask, -1000000.0)

        if do_eval:
            a = torch.argmax(qs).view([])
        elif self.epsilon is None:
            a = torch.argmax(qs).view([])
        elif np.random.rand() > self.epsilon:
            a = torch.argmax(qs).view([])
        else:
            a = torch.randint(low=0, high=9, size=(1,)).view([])

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
            'next_state': s_next,
            'next_mask': m_next,
            'achieved': self.env.achieved.detach(),
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
        self.q_module = Value(self._dummy_env.W, use_antigoal=False, like_q=True)
        self.q_target = Value(self._dummy_env.W, use_antigoal=False, like_q=True)
        self.q_target.load_state_dict(self.q_module.state_dict())

    def _make_agent(self):
        return QAgent(epsilon=self.epsilon, env=self.create_env(), q_module=self.q_module)

    def soft_update(self):
        module_pairs = [
            dict(source=self.q_module, target=self.q_target),
        ]
        for pair in module_pairs:
            for p, p_targ in zip(pair['source'].parameters(), pair['target'].parameters()):
                p_targ.data *= self.polyak
                p_targ.data += (1 - self.polyak) * p.data

    def get_next_qs(self, batch):
        q_max_next = torch.max(
            self.q_target(batch['next_state'], batch['goal']).masked_fill(batch['mask'], -1000000.0),
            dim=1
        )[0]
        return q_max_next

    def get_action_qs(self, batch):
        qs = self.q_module(batch['state'], batch['goal'])
        q = qs[torch.arange(batch['action'].shape[0]), batch['action']]
        return q

    def get_icm_loss(self, batch):
        return self.icm(batch['state'], batch['next_state'], batch['action'])


class SiblingRivalryLearner(BaseSiblingRivalryLearner, DistanceLearner):
    def _make_agent_modules(self):
        self.q_module = Value(self._dummy_env.W, self.use_antigoal, like_q=True)
        self.q_target = Value(self._dummy_env.W, self.use_antigoal, like_q=True)
        self.q_target.load_state_dict(self.q_module.state_dict())


class HERLearner(BaseHERLearner, DistanceLearner):
    pass