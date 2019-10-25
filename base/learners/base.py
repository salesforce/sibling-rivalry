import torch
import torch.nn as nn
from base.modules.intrinsic_motivation import IntrinsicMotivationModule


class BaseLearner(nn.Module):
    AGENT_TYPE = 'Base'
    def __init__(self,
                 gamma=0.99,
                 env_params=None,
                 im_params=None,
                 bootstrap_from_early_terminal=True
                 ):
        super().__init__()

        self.gamma = float(gamma)
        assert 0 < self.gamma <= 1.0

        self.env_params = env_params if env_params is not None else {}
        self.im_params = im_params

        self.bootstrap_from_early_terminal = bool(bootstrap_from_early_terminal)

        # VERY IMPORTANT that the training algorithm uses train_steps to track the number of episodes played
        self.train_steps = nn.Parameter(torch.zeros(1))
        self.train_steps.requires_grad = False

        # Create a dummy environment
        self._dummy_env = self.create_env()

        # Make standard modules such as policy, value, critic, etc.
        self._make_agent_modules()

        # Make intrinsic motivation modules (does nothing if no IM is specified)
        if self.im_params is None:
            self.im = None
            self.im_type = None
            self.im_nu = None
            self.im_lambda = None
            self.im_scale = None
        else:
            self.im_nu = self.im_params.get("nu", 0.01)
            self.im_lambda = self.im_params.get("lambda", 0.05)
            self.im_scale = self.im_params.get('scale', False)
            self.im_type = self.im_params.get("type", None)
            self.im = self._make_im_modules()  # This method must return the intrinsic curiosity module
            assert isinstance(self.im, IntrinsicMotivationModule)

        # Create the agent, which interfaces between the environment and policy to collect rollouts
        self.agent = self._make_agent()

        # Things for bookkeeping
        self._ep_summary = []    # Used to track episode statistics
        self._compress_me = []   # Used to hold trajectories for downstream training (trajectory = list of transitions)
        self._batched_ep = None  # Placeholder for the batched version of _compress_me

    def reset(self):
        return

    def save_checkpoint(self, filepath):
        torch.save(self.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint)


    ###### MUST BE SET FOR EVERYTHING ######
    def _make_env(self):
        raise NotImplementedError

    def _make_agent_modules(self):
        raise NotImplementedError

    def _make_agent(self):
        raise NotImplementedError


    ###### MUST BE SET IF USING INTRINSIC MOTIVATION ######
    def _make_im_modules(self):
        raise NotImplementedError

    def get_im_loss(self, batch):
        assert self.im is not None
        return self.im(batch)


    ###### FOR ON-POLICY LEARNERS ######
    @property
    def batch_keys(self):
        return self.agent.batch_keys

    @property
    def no_squeeze_list(self):
        return self.agent.no_squeeze_list

    def get_values(self, batch):
        raise NotImplementedError

    def get_terminal_values(self, batch):
        raise NotImplementedError

    def get_policy_lprobs_and_nents(self, batch):
        raise NotImplementedError


    ###### FOR OFF-POLICY LEARNERS ######
    def episode_summary(self):
        ep = self.curr_ep
        keys = [k for k in ep[0].keys()]
        batched_ep = {}
        for key in keys:
            batched_ep[key] = torch.stack([e[key] for e in ep]).detach()

        _ = self(batched_ep)

        return [float(x) for x in self._ep_summary]

    def transitions_for_buffer(self, training=None):
        ts = []
        for ep in self._compress_me:
            ts += ep
        return ts

    def normalize_batch(self, batch_dict):
        """Apply batch normalization to the appropriate inputs within the batch"""
        return batch_dict

    def soft_update(self):
        raise NotImplementedError

    def get_next_qs(self, batch):
        raise NotImplementedError

    def get_action_qs(self, batch):
        raise NotImplementedError

    def get_policy_loss_and_actions(self, batch):  # For DDPG
        raise NotImplementedError


    ###### GENERAL EPISODE COLLECTION AND RE-LABELING ######
    def _reset_ep_stats(self):
        self._ep_summary = []
        self._compress_me = []
        self._batched_ep = None

    def play_episode(self, reset_dict=None, do_eval=False, **kwargs):
        self._reset_ep_stats()

        if reset_dict is None:
            reset_dict = {}

        self.agent.play_episode(reset_dict, do_eval, **kwargs)
        self.relabel_episode()

    def relabel_episode(self):
        self._compress_me = []
        self._compress_me.append(self.agent.episode)
        self._add_im_reward()

    def _add_im_reward(self):
        if self.im is not None:
            for ep in self._compress_me:
                batched_episode = {key: torch.stack([e[key] for e in ep]) for key in ep[0].keys()}
                surprisals = self.im.surprisal(batched_episode)

                if self.im_scale:
                    self.train()
                    _ = self._im_bn(surprisals.view(-1, 1))
                    self.eval()
                    surprisals = surprisals / torch.sqrt(self._im_bn.running_var[0])

                for e, s in zip(ep, surprisals):
                    e['reward'] += (self.im_nu * s.detach())


    ###### FOR LOGGING ######
    @property
    def curr_ep(self):
        return self.agent.episode

    @property
    def was_success(self):
        return bool(self.agent.env.is_success)

    @property
    def n_steps(self):
        return len(self.curr_ep)

    def fill_summary(self, *values):
        self._ep_summary = [float(self.was_success)] + [v.item() for v in values]


    ###### WILL BE OVERWRITTEN IN THE ALGORITHM DECORATOR ######
    def forward(self, mini_batch):
        raise NotImplementedError