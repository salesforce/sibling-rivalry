from . import on_policy, off_policy

dec_lookup = {
    'ppo': on_policy.ppo_decorator,
    'dqn': off_policy.dqn_decorator,
    'ddpg': off_policy.ddpg_decorator,
}

def decorate(learner, algorithm):

    algorithm_decorator = dec_lookup[algorithm.lower()]

    @algorithm_decorator
    class Learner(learner):
        ALGORITHM = algorithm.upper()
        pass

    return Learner
