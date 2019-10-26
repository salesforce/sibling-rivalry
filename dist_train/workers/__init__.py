import torch.distributed as dist
from dist_train.workers import baseline

off_policy_manager_lookup = {
    'baseline': baseline.OffPolicy,
    'hierarchical': baseline.HierarchicalOffPolicy
}

on_policy_manager_lookup = {
    'baseline': baseline.OnPolicy,
}

ppo_manager_lookup = {
    'baseline': baseline.PPO,
    'hierarchical': baseline.HierarchicalPPO
}

# For listing the current algorithms (see agents/base/algorithm_deecorators/) that belong to each manager group
on_policy_algos = []  # (ignore PPO here; it is unique)
off_policy_algos = ['ddpg', 'dqn']

def synchronous_worker(rank, config, settings):
    """Create a worker to play episodes on a given port and send the results to the trainer"""

    # Create a distributed process so the workers can share gradients and other such things
    dist.init_process_group(
        backend='gloo',
        init_method='tcp://127.0.0.1:43220',
        rank=rank,
        world_size=settings.N
    )
    print('Rank {:02d} worker successfully initiated the distributed process group!'.format(rank), flush=True)

    train_type = config['train_type']

    style = 'hierarchical' if 'hierarchical' in config['learner_type'].lower() else 'baseline'

    # Training is managed according to the PPO set up
    if train_type == 'ppo':
        manager_class = ppo_manager_lookup[style]

    # Doing some on-policy learning algorithm
    elif train_type in on_policy_algos:
        manager_class = on_policy_manager_lookup[style]

    # Doing some off-policy learning algorithm
    elif train_type in off_policy_algos:
        manager_class = off_policy_manager_lookup[style]

    else:
        raise ValueError('Could not associate train_type "{}" with any known training manager'.format(train_type))

    # Create a manager object for this worker
    manager = manager_class(rank, config, settings)

    # Run through however many epochs we're supposed to
    for _ in range(int(settings.dur)):
        manager.do_epoch()
