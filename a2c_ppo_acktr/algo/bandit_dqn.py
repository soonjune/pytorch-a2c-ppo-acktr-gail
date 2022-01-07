import numpy as np
from math import log,sqrt
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class NNLinTS(nn.Module):
    """NN Linear Thompson Sampling Strategy"""
    def __init__(self, model, obs_shape, action_space, nbArms, bandit_dim,reg=1,nu=0.5,sigma=0.5,name='LinTS'):
        super(NNLinTS, self).__init__()
        self.reg = reg
        self.nu = nu
        self.sigma = sigma
        self.nbArms = nbArms
        self.action_space = action_space
        self.context_dim = obs_shape + action_space + nbArms #context dim = |S|+|A|+|K| 521
        self.bandit_dim = bandit_dim # 30
        self.Bandit_Net = BanditNet(self.context_dim, self.bandit_dim)
        self.model = model
        self.b_optimizer = optim.Adam(self.Bandit_Net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08)

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
        
    def get_skip(self, obs, action, num_processes):
        state = self.model.policy.q_net.features_extractor.cnn(obs.to(device).float()).detach() # 1xS
        action = F.one_hot(action.squeeze(), num_classes=self.action_space).float().reshape(num_processes, -1).to(device) # 1xA 
        context = torch.cat((state ,action), dim=-1) # 1x(S+A)
        context = context.repeat(self.nbArms,1) # Kx(S+A)
        v = torch.arange(self.nbArms).repeat(self.nbArms*num_processes,1) # K x K
        arm_idx = torch.repeat_interleave(torch.arange(self.nbArms),num_processes).reshape(-1,1)
        many_hot_arm_idx = (v<=arm_idx).float().to(device) # KxK
        context = torch.cat((context,many_hot_arm_idx),dim=1).to(device) # Kx(S+A+K)
        with torch.no_grad():
            try:
                N = torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
            except ValueError:
                try:
                    N = torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv+0.0001*torch.eye(self.bandit_dim, device=device)) 
                except ValueError:
                    N = torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv+0.001*torch.eye(self.bandit_dim, device=device))
            theta_tilda = N.sample() # d
            theta_tilda = torch.as_tensor(theta_tilda).to(device)
            z = self.Bandit_Net(context) # Kxd
            reward_tilda = torch.matmul(z,theta_tilda) # Kx1
            reward_tilda = reward_tilda.reshape(self.nbArms,-1) # k x 1
            chosen_arm = torch.argmax(reward_tilda, 0).cpu().detach().numpy()
        return chosen_arm

    def update_banditnet(self, obs, actions, extend_length, target_rewards, batch_size):
        self.Bandit_Net.train()
        with torch.no_grad():
            states = self.model.policy.q_net.features_extractor.cnn(obs.to(device).float()).detach() # BxS
        extend_length = extend_length.view(-1, 1) # Bx1
        actions = F.one_hot(actions.squeeze(), num_classes=self.action_space).float().reshape(batch_size, -1).to(device) # BxA
        v = torch.arange(self.nbArms).repeat(batch_size,1) # BxK
        many_hot_arm_idx = (v <= extend_length).float().to(device)  # BxK     
        contexts = torch.cat((states, actions, many_hot_arm_idx), dim=1).to(device) # Bx(D+A+N)     64x521
        z = self.Bandit_Net(contexts) # Bxd
        '''N=torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
        theta_tilda=N.sample()
        theta_tilda=torch.reshape(torch.as_tensor(theta_tilda), (-1,1)).to(device) # dx1
        reward_tilda=torch.matmul(z,theta_tilda).to(device) '''#Bx1
        mean_reward = torch.matmul(z,self.thetaLS).to(device) # Bx1
        target_rewards = torch.reshape(target_rewards, (-1,1)).to(device)
        bandit_loss = F.mse_loss(mean_reward, target_rewards.detach())
        # Optimize the model
        self.b_optimizer.zero_grad()
        bandit_loss.backward()
        for param in self.Bandit_Net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.b_optimizer.step()

    def update_TS(self, obs, actions, extend_length, target_rewards, batch_size):
        with torch.no_grad():
            states = self.model.policy.q_net.features_extractor.cnn(obs.to(device).float()).detach() # BxS
        extend_length = extend_length.view(-1, 1) # Bx1
        actions = F.one_hot(actions.squeeze(), num_classes=self.action_space).float().reshape(batch_size, -1).to(device) # BxA
        v = torch.arange(self.nbArms).repeat(batch_size,1) # BxK
        many_hot_arm_idx = (v <= extend_length).float().to(device)  # BxK     
        contexts = torch.cat((states, actions, many_hot_arm_idx), dim=1).to(device) # Bx(D+A+N)     64x521
        z = self.Bandit_Net(contexts) # Bxd

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

    def name(self):
        return self.strat_name


