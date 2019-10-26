# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

learner_lookup = {}

from agents.maze_agents.toy_maze import learners as point_maze_learners
learner_lookup['maze'] = point_maze_learners

try:
    from agents.maze_agents.ant_maze import learners as ant_maze_learners
    learner_lookup['antmaze'] = ant_maze_learners
except:
    print('Cannot import ant maze learners. Skipping.')
    pass

from agents.pixgrid_agents import learners as pixgrid_learners
learner_lookup['pixgrid'] = pixgrid_learners


def agent_classes(agent_type, learner_type, train_type):
    assert agent_type in learner_lookup
    agent_learners = learner_lookup[agent_type]

    learner_tag = learner_type + '_' + train_type.upper()
    assert learner_tag in agent_learners

    return agent_learners[learner_tag]
