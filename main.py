#!/usr/bin/env python
# coding: utf-8

# # To Dos

# - [X] Add requirements.txt
# - [ ] Update ReadMe.md
# - [X] Add Psuedocode(s)
# - [X] Change to handle any general environment case
# - [X] Refactor & organize
# - [X] Explore Hyperparameter space
# - [X] Try CNNs with 1D Conv Layers
# - [X] Look into conducting fair experiments/comparisons (incl. evaluation metrics, etc)
# - [X] Try normalizing both returns and values
# - [X] Try continuous actions
# - [X] Try different states 
# - [ ] Try different RL algorithms (own & stable-baselines3) e.g. PPO , etc.
# - [X] Modularize Project
# - [X] Automate everything
# - [X] Try a model-based controller 
# - [X] Try other model-based controllers (e.g. nonlinear, MPC, etc)
# - [ ] Containarize Project
# - [X] Achieve smooth RL actions
# - [ ] Achieve stable RL controller

# # Imports

# In[1]:

#Project
from train import train
from test import testing
import numpy as np
import matplotlib.pyplot as plt
import getopt, sys
from utils import args_help, hyperopt, plot_ca_states

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# # Inputs

# In[4]:


#Environment
env_ids=list(range(2))
env_ids_te=list(range(3))

#Seed (Set seed for experiment reproducibility)
seed = None

#Hyperparameters
#model
asynch=True #False
learn_std=False
hs=[32,64] #size of hidden layer
action_std=0.5
PID=(0.25, 0.0, 0.5)
#method
lrs=[1e-3,1e-4] #=alpha #learning rate (for optimizer) #different lr for actor and critic? Critic needs to learn faster than actor
avg=False #reduce loss with mean; else: use "sum"
#algorithm
gamma=0.99 #discounting factor
episodes=25000 #max no. of episodes (epochs) #episode terminates if: |theta|>12 deg, |x|>2.4 OR len(epsidoe)>200
normalize=False
use_entropy=True

#Trainning
e=0.01
n_updates=7000 #None
vf_coef=0.5; ent_coef=0.001 #entropy regularization parameters
clip_grads=False; max_norm=0.5 #Grad clipping params
l2_critic_loss=False 

#Validation
val_eps=100

#Testing
test_episodes=50
vis=False
trim=250 #None

#Misc.
figsize=(16,8)
t_limit=60
filtered=None
test_only=False
best_model=None
best_model_sb=None
enforce_smooth=False
plot_only=False


def main(env_ids,env_ids_te,hs,lrs,gamma=0.99,episodes=25000,e=0.01,val_eps=1000,test_episodes=100,PID=(0.25,0.0,0.5),n_updates=None,
avg=False,normalize=False,use_entropy=True,vis=False,seed=None,figsize=(16,8),trim=None,test_only=False,best_model=None,best_model_sb=None,
vf_coef=0.5,ent_coef=0.001,max_norm=0.5,l2_critic_loss=False,t_limit=5*60,action_std=0.5,filtered=None,clip_grads=False, enforce_smooth=False,
learn_std=False,asynch=False,plot_only=False):

    """
    env_ids: training evironment ids
    env_ids_te: testing environment ids
    hs: hidden layer sizes
    lrs: learning rates for the optimizer
    gamma=0.99: discount factor
    episodes=25000: number of training episodes
    e=0.01: a discount factor used in computing the running reward
    val_eps=1000: number of validation episodes
    test_episodes=100: number of test episodes
    PID=(0.25,0.0,0.5): PID controller coefficients (in order)
    n_updates=None: total timesteps to train SB model is n_updates*no. of steps in the environment
    avg=False: whether to use reduction of "mean" when computing the losses (default: "sum")
    normalize=False: whether to normalize the advantage
    use_entropy=True: whether to use entropy to manage exploration of model
    vis=False: whether to visualize validation episodes
    seed=None: the seed value used in the program
    figsize=(16,8): size of the plots
    trim=None: the maximum number of timesteps to display for testing plots
    test_only=False: whether to run the test only with the available saved trained models
    best_model=None: in case of test_only=True, the own model to be tested has to be specified in the form [h,lr,env_id]
    best_model_sb=None: in case of test_only=True, the SB model to be tested has to be specified in the form [h,lr,env_id]
    vf_coef=0.5: coefficient of critic loss in the loss function
    ent_coef=0.001: coefficient of entropy in the loss function
    max_norm=0.5: In case clip_grad=True, max_norm defines the maximum norm beyong which to clip the gradients
    l2_critic_loss=False: whether to use L2 (MSE) loss for critic loss function (default: smooth L1 loss)
    t_limit=5*60: time-box limit (in seconds) to run a validation or test episode for
    action_std=0.5: standard deviation of actor network distribution in case its fixed and not learned
    filtered=None: whether and how to filter signals. values are: None, mov_avg, and savgol
    clip_grads=False: whether or not to apply gradient clipping
    enforce_smooth=False: whether or not to enforce smoothing of forces whithin the environment
    learn_std=False: whether or not to learn the standard deviation of the actor network distribution along with the actions (the means)
    asynch=False: whether or not to use separate networks and losses for the actor and the critic
    plot_only=False: whether or not to plot the control actions and states only with the data already available

    """

    if plot_only:
        plot_ca_states(trim,figsize)
        return
    
    #Train & Validation: Hyperparameter Sweep
    if not test_only:
        #Initis.
        avgrs={}
        avgrs_sb={}
        
        for env_id in env_ids:
            for h in hs:
                for lr in lrs:
                    print(f"Training & Validating on env={env_id} and with h={h} & alpha={lr} ... \n")
                    avgr,avgr_sb=train(env_id,h,lr,gamma,episodes,e,n_updates,val_eps,avg,normalize,use_entropy,seed,figsize,vf_coef,
                    ent_coef,max_norm,l2_critic_loss,t_limit,action_std,filtered,clip_grads,enforce_smooth,learn_std,asynch)
                    avgrs[avgr]=[h,lr,env_id]
                    avgrs_sb[avgr_sb]=[h,lr,env_id]

        #Get best models according to avg reward in validation 
        best_model,best_model_sb=hyperopt(avgrs,avgrs_sb,figsize,env_ids)

    #Testing: Generalization Capabilities of the best models
    for env_id in env_ids_te:
        print(f"Testing on test env={env_id} ... \n") 
        testing(env_id,gamma,test_episodes,best_model,best_model_sb,PID,vis,figsize,trim,t_limit,action_std,filtered,enforce_smooth,
        normalize,learn_std,asynch)

if __name__ == "__main__":

    #Update inputs with command-line values:
    execute=True
    s_opts="hi:I:s:d:a:m:g:E:n:u:e:U:v:t:V:T:f:o:B:S:P:c:C:N:l:L:A:F:G:O:M:H:p:"
    l_opts=["help","env_ids=","env_ids_te=","seed=","hs=","lrs=","avg=","gamma=","episodes=","normalize=","use_entropy=","e=","n_updates=",
    "val_eps=","test_episodes=","vis=","trim=","figsize=","test_only=", "best_model=","best_model_sb=","PID=","vf_coef=","ent_coef=",
    "max_norm=","l2_critic_loss=","t_limit=","action_std=","filtered=","clip_grads=","enforce_smooth=","learn_std=","asynch=","plot_only="]
    arg_ls=sys.argv[1:]
    args, vals=getopt.getopt(arg_ls,s_opts,l_opts)
    
    for arg, val in args:
        if arg in ("-h","--help"):
            args_help(s_opts,l_opts)
            execute=False
        elif arg[1:] in s_opts:
            idx=int(s_opts.index(arg[1:])/2)
            assign=l_opts[idx][:-1]+val
            exec(assign)            
        elif arg[2:]+"=" in l_opts:
            assign=arg[2:]+"="+val
            exec(assign)
        else:
            print(f"Incorrect Argument/Usage {arg}{val}... \n")
            args_help(s_opts,l_opts)
            execute=False

    if execute:
        main(env_ids,env_ids_te,hs,lrs,gamma,episodes,e,val_eps,test_episodes,PID,n_updates,avg,normalize,use_entropy,vis,seed,figsize,trim,test_only,
        best_model,best_model_sb,vf_coef,ent_coef,max_norm,l2_critic_loss,t_limit,action_std,filtered,clip_grads,enforce_smooth,learn_std,asynch,plot_only)