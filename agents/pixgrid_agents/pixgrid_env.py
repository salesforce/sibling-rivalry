# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

import torch
import numpy as np


class Env:
    def __init__(self, w=11, n=50, ahw=1, scale_dist=True, hardmode=False, ignore_reset_start=False):
        self.W = w
        self.n = n
        self.ahw = ahw
        self.scale_dist = scale_dist
        self.hardmode = hardmode
        self._ignore_reset_start = bool(ignore_reset_start)

        self._state = dict(s0=None, state=None, goal=None, n=None, done=None)
        self._goal_gen_steps = self.n // (3 if self.hardmode else 6)

        drs, dcs = [], []
        for a in range(self.n_actions):
            dr, dc = self._action2shift(torch.LongTensor([a]))
            drs.append(dr)
            dcs.append(dc)
        self._drs = torch.stack(drs).view(-1)
        self._dcs = torch.stack(dcs).view(-1)

        self.dist_threshold = 0

    @property
    def dist_scale(self):
        return 1.0#max(1.0, float(self._goal_gen_steps) ** 0.81)

    @property
    def n_actions(self):
        return (1 + (2 * self.ahw)) ** 2

    @property
    def state(self):
        return self._state['state'].view(2, self.W, self.W).detach()

    @property
    def goal(self):
        return self._state['goal'].view(1, self.W, self.W).detach()

    @property
    def achieved(self):
        return self.goal if self.is_success else self.state[:1]

    @staticmethod
    def state_loc(state):
        r, c = state[1].nonzero()[0]
        return r, c

    @property
    def loc(self):
        return self.state_loc(self._state['state'])

    @property
    def is_done(self):
        return bool(self._state['done'])

    @property
    def is_success(self):
        d = self.dist(self.goal, self.state)
        return d == 0.0

    @property
    def next_phase_reset(self):
        return {'state': self._state['s0'], 'goal': self.achieved}

    @property
    def sibling_reset(self):
        return {'state': self._state['s0'].detach(), 'goal': self.goal}

    def action_mask(self, state=None):
        if state is None:
            r, c = self.loc
        else:
            r, c = self.state_loc(state)
        new_r = r + self._drs
        new_c = c + self._dcs
        r_valid = (new_r >= 0) * (new_r < self.W)
        c_valid = (new_c >= 0) * (new_c < self.W)
        valid = r_valid * c_valid
        return torch.logical_not(valid.view(-1).detach())
        # return 1 - valid.view(-1).detach()

    def dist(self, goal, outcome):
        return torch.abs(goal[0] - outcome[0]).sum().type(torch.float32) / (self.dist_scale if self.scale_dist else 1)

    def _gen_goal(self, g0=None):
        toggle_action = ((1 + (2 * self.ahw)) * self.ahw) + self.ahw
        toggle_action = torch.LongTensor([toggle_action])
        if g0 is None:
            g = torch.zeros(2, self.W, self.W, requires_grad=False)
            g[1, np.random.randint(0, self.W), np.random.randint(0, self.W)] = 1
        else:
            g = g0 * torch.ones_like(g0)
        n_ = 0
        n_left = 0
        while n_ < self._goal_gen_steps:
            if n_left <= 0:
                # Choose a duration
                n_left = np.random.randint(low=1, high=self.n // 10)

                # Choose a direction
                action_mask = self.action_mask(state=g)
                while True:
                    action = torch.randint(high=self.n_actions, size=(1,))
                    if action_mask[action] == 0:
                        dr, dc = self._action2shift(action)
                        if dr != 0 or dc != 0:
                            break

            # Apply the action and toggle
            g = self.new_state(action, state=g)
            if not self.hardmode:
                g = self.new_state(toggle_action, state=g)
            n_left -= 1
            n_ += 1

        goal = g[:1]
        if torch.sum(goal) == 0:
            goal = self._gen_goal(g0)

        return goal

    def reset(self, state=None, goal=None):
        if state is None or self._ignore_reset_start:
            state = torch.zeros(2, self.W, self.W, requires_grad=False)
            state[1, np.random.randint(0, self.W), np.random.randint(0, self.W)] = 1
        if goal is None:
            goal = self._gen_goal()

        self._state = {
            's0': state,
            'state': state * torch.ones_like(state),
            'goal': goal,
            'n': 0,
            'done': False
        }

    def step(self, action):
        self._state['state'] = self.new_state(action)
        self._state['n'] += 1
        self._state['done'] = (self._state['n'] >= self.n) or self.is_success

    def _action2shift(self, action):
        dr = (action % (1 + (2 * self.ahw))) - self.ahw
        dc = (action // (1 + (2 * self.ahw))) - self.ahw
        return dr, dc

    def new_state(self, action, state=None):
        if state is None:
            state = self._state['state']
        state = state * torch.ones_like(state)

        r, c = self.state_loc(state)
        dr, dc = self._action2shift(action)

        if self.hardmode:
            # Update the one-hot location
            r = torch.clamp(r + dr, 0, self.W - 1)
            c = torch.clamp(c + dc, 0, self.W - 1)
            state[1] *= 0
            state[1, r, c] += 1

            # AND toggle the state at the new location
            state[0, r, c] = 1 - state[0, r, c]

        else:
            if dr == 0 and dc == 0:
                # Toggle the state at the new location
                state[0, r, c] = 1 - state[0, r, c]

            else:
                r = torch.clamp(r + dr, 0, self.W - 1)
                c = torch.clamp(c + dc, 0, self.W - 1)

                # Update the one-hot location
                state[1] *= 0
                state[1, r, c] += 1

        return state