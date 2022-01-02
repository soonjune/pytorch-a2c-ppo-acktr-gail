import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tt(ts):
    """
    Helper Function to cast observation to correct type/device
    """
    if device.type == "cuda":
        return Variable(ts.float().cuda(), requires_grad=False)
    else:
        return Variable(ts.float(), requires_grad=False)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            self.num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, self.num_outputs)
        elif action_space.__class__.__name__ == "Box":
            self.num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, self.num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            self.num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, self.num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        self.hidden_size = hidden_size
        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


class NatureTQN(nn.Module):
    """
    Network to learn the skip behaviour using the same architecture as the original DQN but with additional context.
    The context is expected to be the chosen behaviour action on which the skip-Q is conditioned.
    This Q function is expected to be used solely to learn the skip-Q function
    """

    def __init__(self, env, in_channels=4, num_actions=10, skip_dim=18):
        """
        :param in_channels: number of channel of input. (how many stacked images are used)
        :param num_actions: action values
        """
        super(NatureTQN, self).__init__()
        if env.observation_space.shape[-1] == 84:  # hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        elif env.observation_space.shape[-1] == 42:  # hack
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=4, stride=2)
        else:
            raise ValueError("Check state space dimensionality. Expected nx42x42 or nx84x84. Was:",
                             env.observation_space.shape)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.skip = nn.Linear(1, num_actions)  # Context layer

        self.fc4 = nn.Linear(7 * 7 * 64 + num_actions, 512)  # Combination layer
        self.fc5 = nn.Linear(512, skip_dim)  # Output

        self.num_envs = env.num_envs

    def forward(self, x, action_val=None):
        # Process input image
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Process behaviour context
        x_ = F.relu(self.skip(action_val))

        # Combine both streams
        x = F.relu(self.fc4(
            torch.cat([x.reshape(self.num_envs, -1), x_], 1)))  # This layer concatenates the context and CNN part
        return self.fc5(x)

class TempoRLPolicy(nn.Module):
    def __init__(self, env, obs_shape, action_space, base=None, base_kwargs=None, skip_dim=30):
        super(TempoRLPolicy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            self.num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, self.num_outputs)
            self.skip_Q = NatureTQN(env, 4, num_actions=self.num_outputs, skip_dim=skip_dim)
        elif action_space.__class__.__name__ == "Box":
            self.num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, self.num_outputs)
            self.skip_Q = NatureTQN(env, 4, num_actions=self.num_outputs, skip_dim=skip_dim)
        elif action_space.__class__.__name__ == "MultiBinary":
            self.num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, self.num_outputs)
            self.skip_Q = NatureTQN(env, 4, num_actions=self.num_outputs, skip_dim=skip_dim)
        else:
            raise NotImplementedError
        
        self.skip_optimizer = torch.optim.Adam(self.skip_Q.parameters())

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs
    
    def get_skip(self, states, actions):
        return self.skip_Q(tt(states), tt(actions)).cpu().detach().numpy()

    def train_skip(self, replay_buffer, batch_size=100):
        """
        Train the skip network
        """
        # Sample replay buffer
        state, action, skip, next_state, reward, not_done = replay_buffer.sample(batch_size)

        # Compute the target Q value
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done * np.power(self.discount, skip + 1) * target_Q).detach()

        # Get current Q estimate
        current_Q = self.skip_Q(torch.cat([state, action], 1)).gather(1, skip.long())

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)

        # Optimize the critic
        self.skip_optimizer.zero_grad()
        critic_loss.backward()
        self.skip_optimizer.step()


    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
    


class BanditNet(nn.Module):
    def __init__(self, context_dim, bandit_dim, hidden_size=128):
        super(BanditNet, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Linear(context_dim, hidden_size)), nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU())
        self.bandit_linear = init_(nn.Linear(hidden_size, bandit_dim))
        self.train()

    def forward(self, x):
        x = self.main(x)
        return self.bandit_linear(x)


class Bandit_Policy(Policy):
    def __init__(self, obs_shape, action_space, nbArms, bandit_dim, base=None, reg=1,sigma=0.5, nu=0.5, base_kwargs=None):
        super(Bandit_Policy, self).__init__(obs_shape, action_space, base=base, base_kwargs=base_kwargs)
        self.reg = reg
        self.nu = nu
        self.sigma = sigma
        self.nbArms = nbArms
        self.context_dim = self.base.hidden_size + self.num_outputs + nbArms #context dim = |S|+|A|+|N| 521
        self.bandit_dim = bandit_dim # 30
        self.Bandit_Net = BanditNet(self.context_dim, self.bandit_dim)

        self.clear()

    def clear(self):
        with torch.no_grad():
            # initialize the design matrix, its inverse, 
            # the vector containing the sum of r_s*x_s and the least squares estimate
            self.t=1
            self.Design=[]
            self.DesignInv=[]
            self.Vector=[]
            self.thetaLS=[]
            self.Design=self.reg*torch.eye(self.bandit_dim, device=device)  # dxd
            self.DesignInv=(1/self.reg)*torch.eye(self.bandit_dim, device=device)
            self.Vector=torch.zeros((self.bandit_dim,1), device=device)
            self.thetaLS=torch.zeros((self.bandit_dim,1), device=device) # regularized least-squares estimate
    
    def get_skip(self, inputs, rnn_hxs, masks, action, num_processes, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)

        state =  actor_features # 16xS
        action = F.one_hot(action.squeeze(), num_classes=self.num_outputs)# 16x1

        context = torch.cat((state ,action), dim=-1) # 16x(S+A)
        context = context.repeat(self.nbArms,1) # 16Kx(S+A)

        v = torch.arange(self.nbArms).repeat(self.nbArms*num_processes,1) # 16K x K
        arm_idx = torch.repeat_interleave(torch.arange(self.nbArms),num_processes).reshape(-1,1)
        many_hot_arm_idx = (v<=arm_idx).float().to(device) # 16Kx16K
        context = torch.cat((context,many_hot_arm_idx),dim=1).to(device) #16Kx(S+A+N)

        with torch.no_grad():
            N = torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
            theta_tilda = N.sample() # d
            theta_tilda = torch.as_tensor(theta_tilda).to(device)
            z = self.Bandit_Net(context) # 16Kxd
            reward_tilda = torch.matmul(z,theta_tilda) # 16Kx1
            reward_tilda = reward_tilda.reshape(self.nbArms,-1) # k x 16
            chosen_arm = torch.argmax(reward_tilda, 0).cpu().detach().numpy()
        return chosen_arm