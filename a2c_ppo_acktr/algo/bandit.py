import numpy as np
from math import log,sqrt
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import math 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BanditNet(nn.Module):
    def __init__(self, state_dim, action_dim, non_linearity=F.relu, hidden_dim=50):
        super(BanditNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self._non_linearity = non_linearity

    def forward(self, x):
        x = self._non_linearity(self.fc1(x))
        x = self._non_linearity(self.fc2(x))
        return self.fc3(x)

class NNLinTS:
    """NN Linear Thompson Sampling Strategy"""
    def __init__(self,nbArms,dimension,nu,bandit_dim,bandit_generator=None,reg=1,sigma=0.5,name='LinTS'):
        self.reg=reg
        self.nbArms=nbArms  # K
        self.dimension=dimension  # d
        self.nu=nu
        self.sigma=sigma
        self.strat_name=name
        self.bandit_generator=bandit_generator
        self.latent_space=bandit_dim
        self.Bandit_Net=BanditNet(dimension, self.latent_space).to(device)
        self.optimiser = optim.Adam(self.Bandit_Net.parameters(), lr=0.001, eps=1.5e-4)
        self.FloatTensor = torch.cuda.FloatTensor if device else torch.FloatTensor
        self.LongTensor = torch.cuda.LongTensor if device else torch.LongTensor

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
            self.Design=self.reg*torch.eye(self.latent_space, device=device)  # dxd
            self.DesignInv=(1/self.reg)*torch.eye(self.latent_space, device=device)
            self.Vector=torch.zeros((self.latent_space,1), device=device)
            self.thetaLS=torch.zeros((self.latent_space,1), device=device) # regularized least-squares estimate
        
    def chooseArmToPlay(self, state, action):
        state = Variable(self.FloatTensor(state)) # 1xD
        action = Variable(self.FloatTensor(action)).reshape(1,-1) 
        context = torch.cat((action, state), dim=-1) # 1x(D+1)
        context = context.repeat(self.nbArms,1) # Kx(D+1)
        arm_idx = (torch.arange(self.nbArms).reshape(-1,1)+1)/self.nbArms # Kx1
        #arm_idx = (arm_idx*2-1) # 0~1 -> -1~1
        arm_idx = arm_idx.float().to(device) # Kx1 #?? why not change the last i
        context = torch.cat((context,arm_idx),dim=1).to(device) #Kx(D+1+N)
        with torch.no_grad():
            N=torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
            theta_tilda=N.sample() # d
            theta_tilda=torch.as_tensor(theta_tilda).to(device)
            z=self.Bandit_Net(context) # Kxd
            reward_tilda=torch.matmul(z,theta_tilda).squeeze() # K
            chosen_arm = torch.argmax(reward_tilda, 0).detach().item()
        return chosen_arm

    def update(self, states, actions, extend_length, target_rewards):
        self.Bandit_Net.train()
        states = Variable(self.FloatTensor(states)) # BxD
        #print(extend_length)
        actions = Variable(self.FloatTensor(actions)).view(-1, 1)
        extend_length = Variable(self.FloatTensor(extend_length)).view(-1, 1) # Bx1
        #extend_length = (extend_length*2-1) # 0~1 -> -1~1
        target_rewards = torch.reshape(target_rewards, (-1,1)).to(device)
        contexts = torch.cat(( actions, states, extend_length), dim=1).to(device) # Bx(D+1+N)        
        z = self.Bandit_Net(contexts) # Bxd
        N=torch.distributions.multivariate_normal.MultivariateNormal(self.thetaLS.view(-1),(self.nu*self.nu)*self.DesignInv)
        theta_tilda=N.sample()
        theta_tilda=torch.reshape(torch.as_tensor(theta_tilda), (-1,1)).to(device) # dx1
        reward_tilda=torch.matmul(z,theta_tilda).to(device) #Bx1
        bandit_loss = F.mse_loss(reward_tilda, target_rewards.detach())
        # Optimize the model
        self.optimiser.zero_grad()
        bandit_loss.backward()
        for param in self.Bandit_Net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimiser.step()

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
        return bandit_loss

    def name(self):
        return self.strat_name


