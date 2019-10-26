# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import torch.nn as nn
from torch.distributions import Categorical


class StochasticPolicy(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = int(w)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            #nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            #nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            #nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
        )
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )

        self.softmax_out = nn.Softmax(dim=1)

    def action_stats(self, s, g, mask):
        """Get the action distribution statistics"""
        x = torch.cat([s, g], dim=1)
        x = self.conv_layers(x)
        x = torch.max(torch.max(x, -1)[0], -1)[0]
        x = self.fc_layers(x)
        x = x.masked_fill(mask, -10000.0)
        return self.softmax_out(x)

    def forward(self, s, g, mask, greedy=False, action=None):
        """Produce an action"""
        action_probs = self.action_stats(s, g, mask)
        m = Categorical(action_probs)

        # Sample. (Otherwise, evaluate the action provided as argument)
        if action is None:
            if greedy:
                action = action_probs.argmax().view(-1)
            else:
                action = m.sample()

        n_ents = -m.entropy()
        lprobs = m.log_prob(action)
        return action, lprobs, n_ents


class Value(nn.Module):
    def __init__(self, w, use_antigoal, like_q=False):
        super().__init__()
        self.w = int(w)
        self.use_antigoal = use_antigoal
        self.like_q = like_q

        self.conv_layers = nn.Sequential(
            nn.Conv2d((4 if self.use_antigoal else 3), 32, 3, padding=1),
            #nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            #nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            #nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
        )
        self.fc_layers = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9 if self.like_q else 1),
        )

    def q_no_grad(self, s, g, ag=None):
        for p in self.parameters():
            p.requires_grad = False

        q = self(s, g, ag)

        for p in self.parameters():
            p.requires_grad = True

        return q

    def forward(self, s, g, ag=None):
        if self.use_antigoal:
            x = torch.cat([s, g, ag], dim=1)
        else:
            x = torch.cat([s, g], dim=1)
        x = self.conv_layers(x)
        x = x.mean(dim=-1).mean(dim=-1)
        x = self.fc_layers(x)
        return x if self.like_q else x.view(-1)


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self):
        super().__init__()

        self.action_embedding = nn.Embedding(9, 8)

        self.state_embedding_layers = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
        )
        self.inverse_model_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
        )
        self.inv_model_loss = nn.CrossEntropyLoss()

        self.forward_model_layers = nn.Sequential(
            nn.Linear(64+8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

    def embed_state(self, state):
        """Get the action distribution statistics"""
        x = self.state_embedding_layers(state)
        x = torch.max(torch.max(x, -1)[0], -1)[0]
        return self.normalize(x)

    @staticmethod
    def normalize(x):
        return x / torch.sqrt(torch.pow(x, 2).sum(dim=-1, keepdim=True))

    def surprisal(self, state, next_state, action):
        """Compute surprisal for intrinsic motivation"""
        cat_states = torch.cat([state, next_state], dim=0)
        cat_states_emb = self.embed_state(cat_states)
        state_emb = cat_states_emb[:state.shape[0]]
        next_state_emb = cat_states_emb[state.shape[0]:]

        action_emb = self.action_embedding(action)

        next_state_emb_hat = self.normalize(self.forward_model_layers(torch.cat([state_emb, action_emb], dim=1)))

        return torch.mean(torch.pow(next_state_emb_hat - next_state_emb, 2), dim=1).detach()

    def forward(self, state, next_state, action):
        """Compute terms for intrinsic motivation via surprisal (inlcuding losses and surprise)"""
        cat_states = torch.cat([state, next_state], dim=0)
        cat_states_emb = self.embed_state(cat_states)
        state_emb = cat_states_emb[:state.shape[0]]
        next_state_emb = cat_states_emb[state.shape[0]:]

        action_logits = self.inverse_model_layers(torch.cat([state_emb, next_state_emb], dim=1))
        inv_loss = self.inv_model_loss(action_logits, action).mean()

        action_emb = self.action_embedding(action)
        next_state_emb_hat = self.normalize(self.forward_model_layers(torch.cat([state_emb, action_emb], dim=1)))
        fwd_loss = torch.mean(torch.pow(next_state_emb_hat - next_state_emb.detach(), 2))

        return inv_loss + fwd_loss


class RandomNetworkDistillation(nn.Module):
    def __init__(self):
        super().__init__()

        self.random_network = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
        )

        self.distillation_network = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            # nn.BatchNorm2d(32, affine=False, momentum=0.01),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
        )

    def embed_feature_map(self, feature_map):
        """Get the action distribution statistics"""
        x = torch.max(torch.max(feature_map, -1)[0], -1)[0]
        return self.normalize(x)

    @staticmethod
    def normalize(x):
        return x / torch.sqrt(torch.pow(x, 2).sum(dim=-1, keepdim=True))

    def surprisal(self, state, next_state, action):
        """Compute surprisal for intrinsic motivation"""
        r_state_emb = self.embed_feature_map(self.random_network(next_state))
        d_state_emb = self.embed_feature_map(self.distillation_network(next_state))
        return torch.mean(torch.pow(r_state_emb - d_state_emb, 2), dim=1).detach()

    def forward(self, state, next_state, action):
        """Compute losses for intrinsic motivation via surprisal (inlcuding losses and surprise)"""
        r_state_emb = self.embed_feature_map(self.random_network(next_state)).detach()
        d_state_emb = self.embed_feature_map(self.distillation_network(next_state))
        return torch.mean(torch.pow(r_state_emb - d_state_emb, 2))
