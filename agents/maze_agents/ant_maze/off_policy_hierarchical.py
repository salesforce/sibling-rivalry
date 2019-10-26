# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np
from torch.distributions import Normal
from base.actors.base import BaseHierarchicalActor
from base.learners.distance import BaseDistanceLearner, BaseSiblingRivalryLearner
from base.learners.her import BaseHERLearner
from agents.maze_agents.modules import Policy, Critic, StochasticPolicy, Value
from agents.maze_agents.ant_maze.env import Env


class Agent(BaseHierarchicalActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.noise is not None
        assert self.epsilon is not None

    def _make_modules(self, policy):
        self.policy = policy

    @property
    def rollout(self):
        states = torch.stack([e['state'] for e in self.episode] + [self.episode[-1]['next_state']]).data.numpy()
        xs = states[:, 0]
        ys = states[:, 1]
        return [xs, ys]

    def step(self, do_eval=False):
        s = self.env.state
        g = self.env.goal
        a = self.policy(s.view(1, -1), g.view(1, -1)).view(-1)

        if not do_eval:
            if np.random.rand() < self.epsilon:
                a = np.random.uniform(
                    low=-self.policy.a_range,
                    high=self.policy.a_range,
                    size=(self.policy.action_size,)
                )
                a = torch.from_numpy(a.astype(np.float32))
            else:
                z = Normal(torch.zeros_like(a), torch.ones_like(a) * self.noise * self.policy.a_range)
                a = a + z.sample()
        a = torch.clamp(a, -self.policy.a_range, self.policy.a_range)

        self.lo_rollout(goal_hi=a + self.env.achieved, do_eval=do_eval)

        complete = float(self.env.is_success) * torch.ones(1)
        terminal = float(self.env.is_done) * torch.ones(1)
        s_next = self.env.state
        r = -1 * torch.ones(1)

        self.episode.append({
            'state': s,
            'goal': g,
            'achieved': self.env.achieved,
            'action': a,
            'next_state': s_next,
            'terminal': terminal.view([]),
            'complete': complete.view([]),
            'reward': r.view([]),
        })

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
        self.policy = Policy(self._dummy_env, 128, a_range=5, action_size=2)
        self.p_target = Policy(self._dummy_env, 128, a_range=5, action_size=2)
        self.p_target.load_state_dict(self.policy.state_dict())

        self.q_module = Critic(self._dummy_env, 128, a_range=5, action_size=2)
        self.q_target = Critic(self._dummy_env, 128, a_range=5, action_size=2)
        self.q_target.load_state_dict(self.q_module.state_dict())

        self.policy_lo = StochasticPolicy(self._dummy_env, 256, goal_size=2)
        self.v_module_lo = Value(self._dummy_env, 256, goal_size=2, use_antigoal=False)

    def _make_agent(self):
        return Agent(env=self.create_env(), policy_lo=self.policy_lo, value_lo=self.v_module_lo,
                     noise=self.noise, epsilon=self.epsilon,
                     policy=self.policy, **self._hierarchical_agent_kwargs)

    def soft_update(self):
        module_pairs = [
            dict(source=self.q_module, target=self.q_target),
            dict(source=self.policy, target=self.p_target),
        ]
        for pair in module_pairs:
            for p, p_targ in zip(pair['source'].parameters(), pair['target'].parameters()):
                p_targ.data *= self.polyak
                p_targ.data += (1 - self.polyak) * p.data

    def get_next_qs(self, batch):
        next_policy_actions = self.p_target(batch['next_state'], batch['goal'])
        return self.q_target(batch['next_state'], next_policy_actions, batch['goal'], batch.get('antigoal', None))

    def get_action_qs(self, batch):
        return self.q_module(batch['state'], batch['action'], batch['goal'], batch.get('antigoal', None))

    def get_policy_loss_and_actions(self, batch):
        policy_actions = self.policy(batch['state'], batch['goal'])
        p_losses = -self.q_target.q_no_grad(batch['state'], policy_actions, batch['goal'], batch.get('antigoal', None))
        p_loss = p_losses.mean()
        return p_loss, policy_actions

class SiblingRivalryLearner(BaseSiblingRivalryLearner, DistanceLearner):
    AGENT_TYPE = 'HierarchicalSiblingRivalry'

    def _make_agent_modules(self):
        self.policy = Policy(self._dummy_env, 128, a_range=5, action_size=2)
        self.p_target = Policy(self._dummy_env, 128, a_range=5, action_size=2)
        self.p_target.load_state_dict(self.policy.state_dict())

        self.q_module = Critic(self._dummy_env, 128, a_range=5, action_size=2, use_antigoal=self.use_antigoal)
        self.q_target = Critic(self._dummy_env, 128, a_range=5, action_size=2, use_antigoal=self.use_antigoal)
        self.q_target.load_state_dict(self.q_module.state_dict())

        self.policy_lo = StochasticPolicy(self._dummy_env, 256, goal_size=2)
        self.v_module_lo = Value(self._dummy_env, 256, goal_size=2, use_antigoal=False)

class HERLearner(BaseHERLearner, DistanceLearner):
    AGENT_TYPE = 'HierarchicalHER'
    pass