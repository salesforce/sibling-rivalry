# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from .base import BaseLearner
import numpy as np


class BaseGridOracleLearner(BaseLearner):
    AGENT_TYPE = 'GridOracle'
    def __init__(self,
                 *args,
                 grid_size=2,
                 rew_per_grid=0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.distance = []
        self.locs_visited = set()

        self.grid_size = grid_size
        assert self.grid_size > 0

        self.rew_per_grid = float(rew_per_grid)
        assert self.rew_per_grid >= 0

    @property
    def was_success(self):
        return bool(self.agent.env.is_success)

    @property
    def n_steps(self):
        return int(len(self.agent.episode))

    @property
    def dist_to_goal(self):
        return float(self.distance[-1])

    @property
    def unique_visitations(self):
        return float(len(self.locs_visited))

    def _reset_ep_stats(self):
        super()._reset_ep_stats()
        self.distance = []
        self.locs_visited = set()

    def _count_visitation(self, coords):
        x, y = coords + 0.5
        discritized_x = int(np.floor(x.item() / self.grid_size))
        discritized_y = int(np.floor(y.item() / self.grid_size))
        self.locs_visited.add((discritized_x, discritized_y))

    def relabel_episode(self):
        self._compress_me = []
        for e in self.agent.episode:
            e['reward'] *= 0
            e['reward'] += e['complete']
            self.distance.append(self._dummy_env.dist(e['next_state'], e['goal']).item())

            # Manage set of visited locs
            self._count_visitation(coords=e['next_state'][:2])

        # Add grid oracle reward to terminal action
        self.agent.episode[-1]['reward'] += self.rew_per_grid * self.unique_visitations

        self._compress_me.append(self.agent.episode)
        self._add_im_reward()

    def fill_summary(self, *values):
        manual_summary = [float(self.was_success), float(self.dist_to_goal), float(self.unique_visitations)]

        for v in values:
            manual_summary.append(v.item())

        self._ep_summary = manual_summary
