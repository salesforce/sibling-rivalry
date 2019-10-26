# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from torch.distributions import Beta
from base.modules.intrinsic_motivation import IntrinsicMotivationModule


class Policy(nn.Module):
    def __init__(self, env, hidden_size, a_range=None, state_size=None, goal_size=None, action_size=None):
        super().__init__()
        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        input_size = self.state_size + self.goal_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
        )

    def forward(self, s, g):
        """Produce an action"""
        return torch.tanh(self.layers(torch.cat([s, g], dim=1)) * 0.005) * self.a_range


class StochasticPolicy(nn.Module):
    def __init__(self, env, hidden_size, a_range=None, state_size=None, goal_size=None, action_size=None):
        super().__init__()
        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        input_size = self.state_size + self.goal_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size * 2),
            nn.Softplus()
        )

    def action_stats(self, s, g):
        x = torch.cat([s, g], dim=1)
        action_stats = self.layers(x) + 1.05 #+ 1e-6
        return action_stats[:, :self.action_size], action_stats[:, self.action_size:]

    def scale_action(self, logit):
        # Scale to [-1, 1]
        logit = 2 * (logit - 0.5)
        # Scale to the action range
        action = logit * self.a_range
        return action

    def action_mode(self, s, g):
        c0, c1 = self.action_stats(s, g)
        action_mode = (c0 - 1) / (c0 + c1 - 2)
        return self.scale_action(action_mode)

    def forward(self, s, g, greedy=False, action_logit=None):
        """Produce an action"""
        c0, c1 = self.action_stats(s, g)
        action_mode = (c0 - 1) / (c0 + c1 - 2)
        m = Beta(c0, c1)

        # Sample.
        if action_logit is None:
            if greedy:
                action_logit = action_mode
            else:
                action_logit = m.sample()

            n_ent = -m.entropy().mean()
            lprobs = m.log_prob(action_logit)
            action = self.scale_action(action_logit)
            return action, action_logit, lprobs, n_ent

        # Evaluate the action previously taken
        else:
            n_ent = -m.entropy().mean(dim=1)
            lprobs = m.log_prob(action_logit)
            action = self.scale_action(action_logit)
            return lprobs, n_ent, action


class Critic(nn.Module):
    def __init__(self, env, hidden_size, use_antigoal=False, a_range=None, state_size=None, goal_size=None, action_size=None):
        super().__init__()
        self.use_antigoal = use_antigoal

        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size
        self.action_size = env.action_size if action_size is None else action_size

        input_size = self.state_size + self.goal_size + self.action_size
        if self.use_antigoal:
            input_size += self.goal_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def q_no_grad(self, s, a, g, ag=None):
        for p in self.parameters():
            p.requires_grad = False

        q = self(s, a, g, ag)

        for p in self.parameters():
            p.requires_grad = True

        return q

    def forward(self, s, a, g, ag=None):
        """Produce an action"""
        if self.use_antigoal:
            return self.layers(torch.cat([s, a, g, ag], dim=1)).view(-1)
        else:
            return self.layers(torch.cat([s, a, g], dim=1)).view(-1)


class Value(nn.Module):
    def __init__(self, env, hidden_size, use_antigoal=False, a_range=None, state_size=None, goal_size=None):
        super().__init__()
        self.use_antigoal = use_antigoal

        self.a_range = env.action_range if a_range is None else a_range
        self.state_size = env.state_size if state_size is None else state_size
        self.goal_size = env.goal_size if goal_size is None else goal_size

        input_size = self.state_size + self.goal_size
        if self.use_antigoal:
            input_size += self.goal_size

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, s, g, ag=None):
        if self.use_antigoal:
            return self.layers(torch.cat([s, g, ag], dim=1)).view(-1)
        else:
            return self.layers(torch.cat([s, g], dim=1)).view(-1)


class IntrinsicCuriosityModule(nn.Module, IntrinsicMotivationModule):
    def __init__(self, env, hidden_size, state_size=None, action_size=None):
        super().__init__()

        self.state_size = env.state_size if state_size is None else state_size
        self.action_size = env.action_size if action_size is None else action_size

        self.state_embedding_layers = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.inverse_model_layers = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size),
        )

        self.forward_model_layers = nn.Sequential(
            nn.Linear(self.action_size + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def normalize(x):
        return x / torch.sqrt(torch.pow(x, 2).sum(dim=-1, keepdim=True))

    def surprisal(self, episode_batch):
        """Compute surprisal for intrinsic motivation"""
        state = episode_batch['state']
        next_state = episode_batch['next_state']
        action = episode_batch['action']
        state_emb = self.normalize(self.state_embedding_layers(state))
        next_state_emb = self.normalize(self.state_embedding_layers(next_state))
        next_state_emb_hat = self.normalize(self.forward_model_layers(torch.cat([state_emb, action], dim=1)))
        return torch.mean(torch.pow(next_state_emb_hat - next_state_emb, 2), dim=1)

    def forward(self, mini_batch):
        """Compute terms for intrinsic motivation via surprisal (inlcuding losses and surprise)"""
        state = mini_batch['state']
        next_state = mini_batch['next_state']
        action = mini_batch['action']
        state_emb = self.normalize(self.state_embedding_layers(state))
        next_state_emb = self.normalize(self.state_embedding_layers(next_state))

        action_hat = self.inverse_model_layers(torch.cat([state_emb, next_state_emb], dim=1))
        inv_loss = torch.mean(torch.pow(action_hat - action, 2))

        next_state_emb_hat = self.normalize(self.forward_model_layers(torch.cat([state_emb, action], dim=1)))
        fwd_loss = torch.mean(torch.pow(next_state_emb_hat - next_state_emb.detach(), 2))

        return inv_loss + fwd_loss


class RandomNetworkDistillation(nn.Module, IntrinsicMotivationModule):
    def __init__(self, env, hidden_size, state_size=None):
        super().__init__()

        self.state_size = env.state_size if state_size is None else state_size

        self.random_network = nn.Sequential(
            nn.Linear(self.state_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.ReLU(),
            nn.Linear(2*hidden_size, hidden_size),
        )

        self.distillation_network = nn.Sequential(
            nn.Linear(self.state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def normalize(x):
        return x / torch.sqrt(torch.pow(x, 2).sum(dim=-1, keepdim=True))

    def surprisal(self, episode_batch):
        """Compute surprisal for intrinsic motivation"""
        next_state = episode_batch['next_state']
        r_state_emb = self.normalize(self.random_network(next_state))
        d_state_emb = self.normalize(self.distillation_network(next_state))
        return torch.mean(torch.pow(r_state_emb - d_state_emb, 2), dim=1).detach()

    def forward(self, mini_batch):
        """Compute losses for intrinsic motivation via surprisal (inlcuding losses and surprise)"""
        next_state = mini_batch['next_state']
        r_state_emb = self.normalize(self.random_network(next_state)).detach()
        d_state_emb = self.normalize(self.distillation_network(next_state))
        return torch.mean(torch.pow(r_state_emb - d_state_emb, 2))
