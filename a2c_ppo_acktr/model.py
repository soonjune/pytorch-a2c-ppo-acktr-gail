import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, obs_shape, action_space, nbArms, bandit_dim, base=None, reg=10, sigma=0.5, nu=0.5, base_kwargs=None):  # test 2.reg 1->10
        super(Bandit_Policy, self).__init__(obs_shape, action_space, base=base, base_kwargs=base_kwargs)
        self.reg = reg
        self.nu = nu
        self.sigma = sigma
        self.nbArms = nbArms
        self.context_dim = self.base.hidden_size + self.num_outputs + nbArms #context dim = |S|+|A|+|K| 521
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

        state =  actor_features.float().detach() # 16xS
        action = F.one_hot(action.squeeze(), num_classes=self.num_outputs).float().reshape(num_processes, -1) # 16xA
        context = torch.cat((state ,action), dim=-1) # 16x(S+A)
        context = context.repeat(self.nbArms,1) # 16Kx(S+A)
        v = torch.arange(self.nbArms).repeat(self.nbArms*num_processes,1) # 16K x K
        arm_idx = torch.repeat_interleave(torch.arange(self.nbArms),num_processes).reshape(-1,1)
        many_hot_arm_idx = (v<=arm_idx).float().to(device) # 16KxK
        context = torch.cat((context,many_hot_arm_idx),dim=1).to(device) #16Kx(S+A+K)
        with torch.no_grad():
            N = torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
            theta_tilda = N.sample() # d
            theta_tilda = torch.as_tensor(theta_tilda).to(device)
            z = self.Bandit_Net(context) # 16Kxd
            reward_tilda = torch.matmul(z,theta_tilda) # 16Kx1
            reward_tilda = reward_tilda.reshape(self.nbArms,-1) # k x 16
            chosen_arm = torch.argmax(reward_tilda, 0).cpu().detach().numpy()
        return chosen_arm

    def bandit_update(self, obs, actions, rnn_hxs, masks, extend_length, target_rewards, batch_size, b_optimizer):
        self.Bandit_Net.train()
        with torch.no_grad():
            value, actor_features, rnn_hxs = self.base(obs, rnn_hxs, masks)

        states = actor_features.float().detach() # BxS
        extend_length = extend_length.view(-1, 1) # Bx1
        actions = F.one_hot(actions.squeeze(), num_classes=self.num_outputs).float().reshape(batch_size, -1) # BxA
        v = torch.arange(self.nbArms).repeat(batch_size,1) # BxK
        many_hot_arm_idx = (v <= extend_length).float().to(device)  # BxK
        target_rewards = torch.reshape(target_rewards, (-1,1)).to(device)
        contexts = torch.cat((states, actions, many_hot_arm_idx), dim=1).to(device) # Bx(D+A+N)     64x521

        z = self.Bandit_Net(contexts) # Bxd
        N=torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
        theta_tilda=N.sample()
        theta_tilda=torch.reshape(torch.as_tensor(theta_tilda), (-1,1)).to(device) # dx1
        reward_tilda=torch.matmul(z,theta_tilda).to(device) #Bx1
        bandit_loss = F.mse_loss(reward_tilda, target_rewards.detach())
        # Optimize the model
        b_optimizer.zero_grad()
        bandit_loss.backward()
        for param in self.Bandit_Net.parameters():
            param.grad.data.clamp_(-1, 1)
        b_optimizer.step()

        # Update theta
        with torch.no_grad():
            self.Design = self.Design + torch.matmul(z.T,z) # (dxB) x (Bxd) -> dxd 
            self.Vector = self.Vector + torch.sum(target_rewards.detach()*z, 0).view(-1,1) # Bx1 * Bxd -> Bxd -> dx1
            # online update of the inverse of the design matrix
            omega=torch.matmul(z,self.DesignInv) # (Bxd) x (dxd) -> Bxd           
            self.DesignInv= self.DesignInv- torch.matmul(omega.T,omega)/(1+torch.trace(torch.matmul(z,omega.T)).item()) # (dxd) / (BxB) -> dxd
            # update of the least squares estimate 
            self.thetaLS = torch.matmul(self.DesignInv,self.Vector) # d
            self.t+=1