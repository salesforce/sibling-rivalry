# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from base.actors.base import BaseActor
from base.learners.distance import BaseDistanceLearner, BaseSiblingRivalryLearner
from base.learners.grid_oracle import BaseGridOracleLearner
from agents.maze_agents.modules import StochasticPolicy, Value, IntrinsicCuriosityModule, RandomNetworkDistillation
from agents.maze_agents.toy_maze.env import Env


class StochasticAgent(BaseActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch_keys = [
            'state', 'next_state', 'goal',
            'action', 'n_ent', 'log_prob', 'action_logit',
            'reward', 'terminal', 'complete',
        ]
        self.no_squeeze_list = []

    def _make_modules(self, policy):
        self.policy = policy

    def step(self, do_eval=False):
        s = self.env.state
        g = self.env.goal
        a, logit, log_prob, n_ent = self.policy(s.view(1, -1), g.view(1, -1), greedy=do_eval)
        a = a.view(-1)
        logit = logit.view(-1)
        log_prob = log_prob.sum()

        self.env.step(a)
        complete = float(self.env.is_success) * torch.ones(1)
        terminal = float(self.env.is_done) * torch.ones(1)
        s_next = self.env.state
        r = -1 * torch.ones(1)

        self.episode.append({
            'state': s,
            'goal': g,
            'action': a,
            'action_logit': logit,
            'log_prob': log_prob.view([]),
            'n_ent': n_ent.view([]),
            'next_state': s_next,
            'terminal': terminal.view([]),
            'complete': complete.view([]),
            'reward': r.view([]),
        })

    @property
    def rollout(self):
        states = torch.stack([e['state'] for e in self.episode] + [self.episode[-1]['next_state']]).data.numpy()
        xs = states[:, 0]
        ys = states[:, 1]
        return [xs, ys]


class DistanceLearner(BaseDistanceLearner):
    def create_env(self):
        return Env(**self.env_params)

    def _make_agent_modules(self):
        self.policy = StochasticPolicy(self._dummy_env, 128)
        self.v_module = Value(self._dummy_env, 128, use_antigoal=False)

    def _make_im_modules(self):
        """Only gets called if 'im_params' config argument is not None (default=None)."""
        im_support = {
            'icm': IntrinsicCuriosityModule,
            'rnd': RandomNetworkDistillation
        }
        if self.im_type.lower() not in im_support:
            raise ValueError('Intrinsic Motivation type {} not recognized. Options are:\n{}'.format(
                self.im_type.lower(),
                '\n'.join([k for k in im_support.keys()])
            ))
        im_class = im_support[self.im_type.lower()]
        return im_class(env=self._dummy_env, hidden_size=128)

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
        log_prob, n_ent, _ = self.policy(
            batch['state'],
            batch['goal'],
            action_logit=batch['action_logit']
        )
        return log_prob.sum(dim=1), n_ent


class SiblingRivalryLearner(BaseSiblingRivalryLearner, DistanceLearner):
    def _make_agent_modules(self):
        self.policy = StochasticPolicy(self._dummy_env, 128)
        self.v_module = Value(self._dummy_env, 128, use_antigoal=self.use_antigoal)

    def _make_agent(self):
        agent = super()._make_agent()
        agent.batch_keys.append('antigoal')
        return agent

class GridOracleLearner(BaseGridOracleLearner, DistanceLearner):
    AGENT_TYPE = 'GridOracle'
    pass