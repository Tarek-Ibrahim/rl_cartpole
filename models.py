#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:
# import numpy as np
#ML Framework (PyTorch)
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal as mvn
from torch.distributions.normal import Normal as gauss_distrib


# # Architecture

# In[3]:

class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size_1, hidden_size_2, device, action_std,normalize,learn_std):
        super().__init__()
        
        self.learn_std=learn_std
        if learn_std:
            output_size*=2
        else:
            self.action_std=action_std

        self.device=device
        last_layer= nn.Tanh() if normalize else nn.Identity()
        
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(),
            nn.Linear(hidden_size_2, output_size),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size_1),
            nn.Tanh(),
            nn.Linear(hidden_size_1, hidden_size_2),
            nn.Tanh(), 
            nn.Linear(hidden_size_2, 1),
            last_layer
        )
    
    def forward(self, x):

        value = self.critic_net(x)
        dist= self.actor_net(x)
        return dist, value

    def actor_net(self,x):

        x=torch.FloatTensor(x).unsqueeze(0).to(self.device)
        if self.learn_std:
            probs = self.actor(x)[:,0] #mean
            self.action_std=torch.exp(-self.actor(x)[:,1]) #std
            # action_std=nn.Sigmoid()(self.actor(x)[:,1])
        else:
            probs=self.actor(x)

        dist=gauss_distrib(probs,self.action_std)

        return dist

    def critic_net(self,x):
        
        x=torch.FloatTensor(x).unsqueeze(0).to(self.device)
        value = self.critic(x)
        return value
    