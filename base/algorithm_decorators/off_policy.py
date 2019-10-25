import torch
from base.learners.base import BaseLearner


def dqn_decorator(partial_agent_class):
    assert issubclass(partial_agent_class, BaseLearner)

    class NewClass(partial_agent_class):
        def __init__(self, epsilon=0.0, polyak=0.95, **kwargs):
            self.epsilon = epsilon
            self.polyak = polyak

            super().__init__(**kwargs)

        def forward(self, mini_batch):
            self.train()

            # Get the q target
            q_next = self.get_next_qs(mini_batch)

            # Bootstrap from early terminal means bootstrap from terminal value when episode didn't complete
            if self.bootstrap_from_early_terminal:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['complete']) * self.gamma * q_next)
            else:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['terminal']) * self.gamma * q_next)

            # Get the Q values associated with the observed transitions
            q = self.get_action_qs(mini_batch)

            # Loss for the q_module
            q_loss = torch.pow(q - q_targ.detach(), 2).mean()

            p_loss = torch.zeros_like(q_loss)
            n_ent = torch.zeros_like(q_loss)

            self.fill_summary(mini_batch['reward'].mean(), q.mean(), q_loss, p_loss, n_ent)

            if self.im is not None:
                q_loss += (self.im_lambda * self.get_im_loss(mini_batch))

            self.eval()

            return q_loss

    return NewClass


def ddpg_decorator(partial_agent_class):
    assert issubclass(partial_agent_class, BaseLearner)

    class NewClass(partial_agent_class):
        def __init__(self, noise=0.0, epsilon=0.0, action_l2_lambda=0.0, polyak=0.95, **kwargs):
            self.noise = float(noise)
            assert 0 <= self.noise

            self.epsilon = float(epsilon)
            assert 0 <= self.epsilon <= 1.0

            self.action_l2_lambda = float(action_l2_lambda)
            assert 0 <= self.action_l2_lambda

            self.polyak = polyak
            assert 0 <= self.polyak <= 1.0

            super().__init__(**kwargs)

        def forward(self, mini_batch):
            # Get the q target
            q_next = self.get_next_qs(mini_batch)

            # Bootstrap from early terminal means bootstrap from terminal value when episode didn't complete
            if self.bootstrap_from_early_terminal:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['complete']) * self.gamma * q_next)
            else:
                q_targ = mini_batch['reward'] + ((1 - mini_batch['terminal']) * self.gamma * q_next)
            # q_targ = torch.clamp(q_targ, *self._q_clamp)

            # Get the Q values associated with the observed transitions
            q = self.get_action_qs(mini_batch)

            # Loss for the q_module
            q_loss = torch.pow(q - q_targ.detach(), 2).mean()

            # We want to optimize the actions wrt their q value (without getting q module gradients)
            p_loss, policy_actions = self.get_policy_loss_and_actions(mini_batch)

            l2 = torch.mean(policy_actions ** 2)
            l2_loss = l2 * self.action_l2_lambda

            self.fill_summary(mini_batch['reward'].mean(), q.mean(), q_loss, p_loss, l2)

            loss = p_loss + q_loss + l2_loss

            if self.im is not None:
                loss += (self.im_lambda * self.get_im_loss(mini_batch))

            return loss

    return NewClass