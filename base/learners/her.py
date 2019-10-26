# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from .distance import BaseDistanceLearner
import torch
import numpy as np

class BaseHERLearner(BaseDistanceLearner):
    AGENT_TYPE = 'HER'
    def __init__(self, k=4, **kwargs):
        self.k = k
        # Switch the default here to NOT use dense reward (by default, this is true HER w/ sparse reward)
        if 'shaped_type' not in kwargs:
            kwargs['shaped_type'] = 'off'
        super().__init__(**kwargs)

        # This requires bootstrapping from non-complete terminal steps
        self.bootstrap_from_early_terminal = True

    def transitions_for_buffer(self, training=None):
        if self.im is not None:
            raise NotImplementedError

        her_transitions = super().transitions_for_buffer(training=training)

        for t in range(len(self.curr_ep)):
            perm_idx = [int(i) for i in np.random.permutation(np.arange(t, len(self.curr_ep)))]
            her_goal_indices = perm_idx[:self.k]
            for idx in her_goal_indices:
                her_goal = self.curr_ep[idx]['achieved']

                new_trans = {k: v.clone().detach() for k, v in self.curr_ep[t].items()}
                new_trans['goal'] = her_goal.detach()

                # This transition makes no sense. Skip it.
                if self._dummy_env.dist(new_trans['state'], new_trans['goal']) <= self._dummy_env.dist_threshold:
                    continue

                # This action completed the HER goal
                if self._dummy_env.dist(new_trans['next_state'], new_trans['goal']) <= self._dummy_env.dist_threshold:
                    new_trans['terminal'] = torch.ones_like(new_trans['terminal'])
                    new_trans['complete'] = torch.ones_like(new_trans['complete'])


                # Re-label with the new HER goal
                r = new_trans['complete'] + self.distance_reward(new_trans)
                new_trans['reward'] *= 0
                new_trans['reward'] += r

                her_transitions.append(new_trans)

        return her_transitions