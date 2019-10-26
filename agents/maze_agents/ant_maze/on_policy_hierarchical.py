# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
from base.actors.base import BaseHierarchicalActor
from base.learners.distance import BaseDistanceLearner, BaseSiblingRivalryLearner
from base.learners.grid_oracle import BaseGridOracleLearner
from agents.maze_agents.modules import StochasticPolicy, Value, IntrinsicCuriosityModule, RandomNetworkDistillation
from agents.maze_agents.ant_maze.env import Env


class StochasticAgent(BaseHierarchicalActor):
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

        self.lo_rollout(goal_hi=a + self.env.achieved, do_eval=do_eval)

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
        states = torch.stack([e['state'] for e in self.episode_full] + [self.episode_full[-1]['next_state']]).data.numpy()
        xs = states[:, 0]
        ys = states[:, 1]
        return [xs, ys]

class DistanceLearner(BaseDistanceLearner):
    AGENT_TYPE = 'HierarchicalDistance'
    def __init__(self, *args,
                 hi_skip=10, entropy_lambda_lo=0.02, n_lo_epochs=1,
                 **kwargs):
        self._hierarchical_agent_kwargs = dict(
            hi_skip=hi_skip, entropy_lambda=entropy_lambda_lo, n_lo_epochs=n_lo_epochs
        )
        super().__init__(*args, **kwargs)

        self._lo_parameters = self.agent._lo_parameters

    def create_env(self):
        return Env(**self.env_params)

    def _make_agent_modules(self):
        self.policy = StochasticPolicy(self._dummy_env, 256, a_range=5, action_size=self._dummy_env.goal_size)
        self.v_module = Value(self._dummy_env, 256, use_antigoal=False)

        self.policy_lo = StochasticPolicy(self._dummy_env, 256)
        self.v_module_lo = Value(self._dummy_env, 256, use_antigoal=False)

    def _make_agent(self):
        return StochasticAgent(env=self.create_env(), policy_lo=self.policy_lo, value_lo=self.v_module_lo,
                               policy=self.policy, **self._hierarchical_agent_kwargs)

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
        return im_class(env=self._dummy_env, hidden_size=128, action_size=2)

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
            batch['state'], batch['goal'],
            action_logit=batch['action_logit']
        )
        return log_prob.sum(dim=1), n_ent

    def get_icm_loss(self, batch):
        return self.icm(batch['state'], batch['next_state'], batch['action'])


class SiblingRivalryLearner(BaseSiblingRivalryLearner, DistanceLearner):
    AGENT_TYPE = 'HierarchicalSiblingRivalry'

    def _make_agent_modules(self):
        self.policy = StochasticPolicy(self._dummy_env, 256, a_range=5, action_size=self._dummy_env.goal_size)
        self.v_module = Value(self._dummy_env, 256, use_antigoal=self.use_antigoal)

        self.policy_lo = StochasticPolicy(self._dummy_env, 256)
        self.v_module_lo = Value(self._dummy_env, 256, use_antigoal=False)

    def _make_agent(self):
        agent = super()._make_agent()
        agent.batch_keys.append('antigoal')
        return agent

class GridOracleLearner(BaseGridOracleLearner, DistanceLearner):
    AGENT_TYPE = 'HierarchicalGridOracle'
    pass