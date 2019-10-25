import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np


class BaseActor(nn.Module):
    def __init__(self, env, noise=None, epsilon=None, **module_kwargs):
        super().__init__()
        self.noise = max(0.0, float(noise)) if noise is not None else noise
        self.epsilon = max(0.0, min(1.0, float(epsilon))) if epsilon is not None else epsilon

        self.env = env
        self._make_modules(**module_kwargs)

        self.episode = []

    def _make_modules(self, **module_kwargs):
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.episode = []

    def play_episode(self, reset_dict={}, do_eval=False):
        self.reset(**reset_dict)
        while not self.env.is_done:
            self.step(do_eval)

    def step(self, do_eval=False):
        raise NotImplementedError

    @property
    def rollout(self):
        return self.env.episode_to_rollout(self.episode)


class BaseHierarchicalActor(nn.Module):
    def __init__(self, env, policy_lo, value_lo, noise=None, epsilon=None,
                 hi_skip=10, entropy_lambda=0.02, n_lo_epochs=1,
                 **module_kwargs
                 ):
        super().__init__()
        self.noise = max(0.0, float(noise)) if noise is not None else noise
        self.epsilon = max(0.0, min(1.0, float(epsilon))) if epsilon is not None else epsilon

        self.env = env
        self.policy_lo = policy_lo
        self.value_lo = value_lo

        self._lo_parameters = list(self.value_lo.parameters()) + list(self.policy_lo.parameters())

        self._make_modules(**module_kwargs)

        self.episode = []
        self.episode_full = []
        self.episode_lo   = []

        self.hi_skip = hi_skip
        self.gamma = 0.98
        self.gae_lambda = 0.95
        self.entropy_lambda = entropy_lambda
        self.n_lo_epochs = int(n_lo_epochs)


        # For doing PPO updates on the low-level stuff
        self.clip_range = 0.2
        self._mini_buffer = {'state': None}
        self._epoch_transitions = {}

        self.n_mini_batches = int(self.hi_skip)
        for n in range(4, self.hi_skip):
            if (self.hi_skip % n) == 0:
                self.n_mini_batches = int(n)
                break

    def _make_modules(self, **module_kwargs):
        raise NotImplementedError

    def reset(self, state=None, goal=None):
        self.env.reset(state, goal)
        self.episode = []
        self.episode_full = []
        self.episode_lo = []

    @property
    def rollout(self):
        states = torch.stack([e['pre_achieved'] for e in self.episode_full] + [self.episode_full[-1]['achieved']]).data.numpy()
        xs = states[:, 0]
        ys = states[:, 1]
        return [xs, ys]

    @property
    def current_horizon(self):
        mb_state = self._mini_buffer['state']
        if mb_state is None:
            return 0
        else:
            return mb_state.shape[0]

    def add_to_mini_buffer(self, batched_episode):
        for k, v in batched_episode.items():
            if self._mini_buffer.get(k, None) is None:
                self._mini_buffer[k] = v.detach()
            else:
                self._mini_buffer[k] = torch.cat([self._mini_buffer[k], v], dim=0)

        curr_horizon = int(self.current_horizon)
        assert all([int(v.shape[0]) == curr_horizon for v in self._mini_buffer.values()])

    def fill_epoch_transitions(self):
        curr_horizon = int(self.current_horizon)
        assert curr_horizon >= self.n_mini_batches
        self._epoch_transitions = {}
        for k, v in self._mini_buffer.items():
            self._epoch_transitions[k] = v.detach()
            self._mini_buffer[k] = None

    def make_epoch_mini_batches(self, normalize_advantage=False):
        sz = [v.shape[0] for v in self._epoch_transitions.values()][0]
        n_total = self.n_mini_batches * (sz // self.n_mini_batches)
        perm_indices = np.random.permutation(sz)[:n_total]
        mb_indices = np.split(perm_indices, self.n_mini_batches)

        mini_batches = []
        for indices in mb_indices:
            this_batch = {k: v[indices] for k, v in self._epoch_transitions.items()}
            if normalize_advantage and 'advantage' in this_batch:
                mb_mean = this_batch['advantage'].mean(dim=0, keepdim=True)
                mb_std = this_batch['advantage'].std(dim=0, keepdim=True)
                this_batch['advantage'] = (this_batch['advantage'] - mb_mean) / mb_std
            mini_batches.append(this_batch)
        return mini_batches

    def distributed_advantage_normalization(self):
        if 'advantage' not in self._epoch_transitions:
            return
        a = self._epoch_transitions['advantage']

        a_sum = a.sum(dim=0)
        a_sumsq = torch.pow(a, 2).sum(dim=0)
        dist.all_reduce(a_sum)
        dist.all_reduce(a_sumsq)

        n = a.shape[0] * torch.ones(1)
        dist.all_reduce(n)
        # n = dist.get_world_size() * self.horizon

        a_mean = a_sum / n
        a_var = (a_sumsq / n) - (a_mean ** 2)
        a_std = torch.pow(a_var, 0.5) + 1e-8

        n_dim = len(a.shape)
        if n_dim == 1:
            self._epoch_transitions['advantage'] = (a - a_mean) / a_std
        elif n_dim == 2:
            self._epoch_transitions['advantage'] = (a - a_mean.view(1, -1)) / a_std.view(1, -1)
        else:
            raise NotImplementedError

    def play_episode(self, reset_dict={}, do_eval=False, optim_lo=None, distributed=True):
        self.reset(**reset_dict)
        while not self.env.is_done:
            self.step(do_eval)

        # Compute the lower-level loss for this episode and do a PPO epoch on the low-level policy/value modules
        if optim_lo is not None:
            optim_lo.zero_grad()

            keys = ['state', 'goal', 'action_logit', 'log_prob', 'cumulative_return', 'advantage']
            batched_ep_lo = {k: torch.stack([e[k] for e in self.episode_lo]).detach() for k in keys}
            self.add_to_mini_buffer(batched_ep_lo)
            self.fill_epoch_transitions()

            if distributed:
                self.distributed_advantage_normalization()

            for _ in range(self.n_lo_epochs):
                for mini_batch in self.make_epoch_mini_batches(normalize_advantage=not distributed):
                    optim_lo.zero_grad()

                    value = self.value_lo(
                        mini_batch['state'],
                        mini_batch['goal']
                    )
                    v_loss = 0.5 * torch.pow(mini_batch['cumulative_return'] - value, 2).mean()

                    log_prob, n_ent, greedy_action = self.policy_lo(
                        mini_batch['state'], mini_batch['goal'],
                        action_logit=mini_batch['action_logit']
                    )
                    e_loss = n_ent.mean()

                    # Defining Loss = - J is equivalent to max J
                    log_prob = log_prob.sum(dim=1)
                    ratio = torch.exp(log_prob - mini_batch['log_prob'])

                    pg_losses1 = -mini_batch['advantage'] * ratio
                    pg_losses2 = -mini_batch['advantage'] * torch.clamp(ratio,
                                                                        1.0 - self.clip_range,
                                                                        1.0 + self.clip_range)
                    p_losses = torch.max(pg_losses1, pg_losses2)
                    p_loss = p_losses.mean()

                    loss = v_loss + p_loss + (self.entropy_lambda * e_loss)
                    loss.backward()

                    if distributed:
                        for p in self._lo_parameters:
                            if p.grad is not None:
                                dist.all_reduce(p.grad.data)
                                p.grad.data /= dist.get_world_size()
                    optim_lo.step()

    def step(self, do_eval=False):
        raise NotImplementedError

    def lo_rollout(self, goal_hi, do_eval=False):
        """Have the low-level policy follow the high-level instruction and compute loss"""
        sub_ep_lo = []

        goal_hi = goal_hi.detach()

        for _ in range(self.hi_skip):
            s = self.env.state
            pre_achieved = self.env.achieved
            v = self.value_lo(s.view(1, -1), goal_hi.view(1, -1))
            a, a_logit, log_prob, n_ent = self.policy_lo(s.view(1, -1), goal_hi.view(1, -1), greedy=do_eval)
            a = a.view(-1)
            a_logit = a_logit.view(-1)
            log_prob = log_prob.sum()

            self.env.step(a.data.numpy())
            complete = float(self.env.is_success) * torch.ones(1)
            terminal = float(self.env.is_done) * torch.ones(1)
            s_next = self.env.state
            achieved = self.env.achieved
            r = -1 * torch.ones(1)

            self.episode_full.append({
                'state': s,
                'goal': self.env.goal,
                'goal_hi': goal_hi,
                'pre_achieved': pre_achieved,
                'action': a,
                'n_ent': n_ent.view([]),
                'next_state': s_next,
                'achieved': achieved,
                'terminal': terminal.view([]),
                'complete': complete.view([]),
                'reward': r.view([]),
            })

            sub_ep_lo.append({
                'state': s,
                'goal': goal_hi,
                'value': v.view([]),
                'action': a,
                'action_logit': a_logit,
                'log_prob': log_prob.view([]),
                'next_state': s_next,
                'reward': -self.env.dist(goal_hi, s_next).view([]),
            })

        last_adv = 0
        for t in reversed(range(self.hi_skip)):
            if t == self.hi_skip - 1:
                next_value = self.value_lo(
                    sub_ep_lo[-1]['next_state'].view(1, -1), sub_ep_lo[-1]['goal'].view(1, -1)
                )
            else:
                next_value = sub_ep_lo[t + 1]['value']

            delta = sub_ep_lo[t]['reward'] + self.gamma * next_value - sub_ep_lo[t]['value']
            sub_ep_lo[t]['advantage'] = (delta + self.gamma * self.gae_lambda * last_adv).detach().view([])
            sub_ep_lo[t]['cumulative_return'] = (sub_ep_lo[t]['advantage'] + sub_ep_lo[t]['value']).detach().view([])
            last_adv = sub_ep_lo[t]['advantage'].detach()

        self.episode_lo += sub_ep_lo