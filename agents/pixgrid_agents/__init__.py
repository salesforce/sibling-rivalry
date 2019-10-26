# Copyright (c) 2019, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: MIT
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/MIT

from base.algorithm_decorators import decorate

learners = {}

def add_to_learners(learner, *algorithms):
    for algorithm in algorithms:
        algo_learner = decorate(learner, algorithm)
        learners[algo_learner.AGENT_TYPE + '_' + algo_learner.ALGORITHM] = algo_learner


###### ON POLICY ############

from .on_policy import DistanceLearner as Learner
add_to_learners(Learner, 'ppo')

from .on_policy import SiblingRivalryLearner as Learner
add_to_learners(Learner, 'ppo')


###### OFF POLICY ############

from .off_policy import DistanceLearner as Learner
add_to_learners(Learner, 'dqn')

from .off_policy import SiblingRivalryLearner as Learner
add_to_learners(Learner, 'dqn')

from .off_policy import HERLearner as Learner
add_to_learners(Learner, 'dqn')