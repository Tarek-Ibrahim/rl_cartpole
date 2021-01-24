#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


#Essentials
import numpy as np
import matplotlib.pyplot as plt

#Funcs
from utils import validate, plot_func
from models import ActorCritic

#Utils
import tqdm
from copy import copy, deepcopy
from scipy.ndimage import uniform_filter1d as mov_avg
from scipy.signal import savgol_filter
import decimal

#Envs.
import gym
import gym_custom

#ML Framework (PyTorch)
import torch
from torch import nn
import torch.optim as optim

#Stable-baselines3
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env


def train(env_id,h,lr,gamma,episodes,e,n_updates,val_eps,avg,normalize,use_entropy,seed,figsize,vf_coef,ent_coef,max_norm,l2_critic_loss,
t_limit,action_std,filtered,clip_grads,enforce_smooth,learn_std,asynch):

    # # Initializations

    #Device
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    #Environment
    max_r_ls=[1,72,72] # max_r_ls=[1,13,np.exp(3)-1]
    env=gym.make("env-v"+str(env_id))
    env_val=gym.make("env-v0")
    check_env(env) #check the env for sb 
    env.env.enforce_smooth=enforce_smooth #env.env.saturate=True
    env_val.env.enforce_smooth=enforce_smooth #env_val.env.saturate=True
    env_val._max_episode_steps=1e12
    max_r_step=max_r_ls[env_id] #max reward per step
    n=env.observation_space.shape[0] #length of state vector (=4) #state is continous #[x,v,theta,omega] #velocities=|R #x=+-4.8 #theta=+-24 degrees
    m=env.action_space.shape[0] #no. of actions #continous actions [-1,1]
        
    # Seed (Set seed for experiment reproducibility)
    if seed is not None:
        env.seed(seed)
        env.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    #Hyperparameters
    #method
    astr="1e"+str(decimal.Decimal(str(lr)).as_tuple().exponent)
    #algorithm
    steps=copy(env._max_episode_steps) #1000 #200 #max no. of steps (per episode) #=max episode length
    #Model
    model=ActorCritic(input_size=n,output_size=m,hidden_size_1=h,hidden_size_2=h,device=device,action_std=action_std,
    normalize=normalize,learn_std=learn_std).to(device) #Own Model Init
    model_sb = A2C("MlpPolicy", env, verbose=0, gamma=gamma, normalize_advantage=normalize,device=device,
                policy_kwargs=dict(net_arch=[dict(pi=[h,h], vf=[h,h])]), 
                n_steps=steps,
                seed=seed,
                max_grad_norm=max_norm,
                vf_coef=vf_coef, ent_coef=ent_coef,
                use_rms_prop=False, #use Adam optimizer
                learning_rate=lr )
    #Method
    if asynch:
        optimizer_critic=optim.Adam(model.critic.parameters(),lr=lr)
        optimizer_actor=optim.Adam(model.actor.parameters(),lr=lr*0.1)
    else:
        optimizer=optim.Adam(model.parameters(),lr=lr) #SGD + momentum + adaptive vector sizess
    loss_func=nn.MSELoss() if l2_critic_loss else nn.SmoothL1Loss(reduction="mean") if avg else nn.SmoothL1Loss(reduction="sum") #Huber loss #more stable/robust to outliers than VE_bar

    #Problem (Considered solved when the average return is greater than or equal to 97.5% of the maximum episode reward over 100 consecutive trials/episodes)
    max_r_ep=max_r_step*steps #max reward per episode
    r_thr=np.ceil(0.95*max_r_ep) #reward threshold
    env._reward_threshold=copy(r_thr)

    #Train Inits.
    running_reward = 0.
    con_r=0.
    max_con=0.
    patience=0
    pat_thr=101
    best_rr=[-1e6,0]
    plot_rr=[]
    old_er=0
    epsilon=[]

    #Misc.
    prog=lambda x: tqdm.trange(x, leave=True)
    folder_own="models/own/"
    folder_sb="models/sb/"
    folder_op_plt="output/results_and_plots/train/"
    env_name=f"ac_env{env_id}_"
    model_name=f"h{h}_alpha{astr}"
    file_name=env_name+model_name
    file_name_own=folder_own+file_name+".pt"
    file_name_sb=folder_sb+file_name


    # # Implementation

    # ## Our Implementation

    # ### Train

    # In[6]:


    t = prog(episodes)
    for episode in t: #Run until solved OR episode>episodes:

        s=env.reset() #Init state #All observations are assigned a uniform random value in [-0.05..0.05]
        episode_r=0. #Init episode reward
        log_probs=[]; values=[]; rewards=[]; returns=[] #Init histories: log(a) [log_probs], values, rewards
        explore_rate=[]
        entropy=0.
        if asynch:
            optimizer_critic.zero_grad()
            optimizer_actor.zero_grad()
        else:    
            optimizer.zero_grad() #[zero the optimizer's grads] #prevents accumlation of gradients (which is only useful in RNNs)
        
        for _ in range(steps): #Run the episode (until step>steps OR termination condition met [raises flag "done"])
            
            if asynch:
                a_probs=model.actor_net(s)
                v=model.critic_net(s)
            else:
                a_probs, v=model(s) #Predict action probabilities ( a_probs=pi(.|s,theta) ) and estimated future rewards from current state (s) ( i.e. state value, V_critic(s) = v_hat(s,w) )
            
            if learn_std: explore_rate.append(a_probs.stddev.squeeze().cpu().detach().numpy().flatten())
            a=a_probs.sample() #sample/draw action (a) from a_probs (a~pi)
            s, r, done, _ =env.step(a.squeeze().cpu().numpy().flatten()) #Take action: Apply action (a) to the environment to get next state (s_dash) and reward (r)
            log_a=a_probs.log_prob(a) #Take log of (a) ( =ln[pi(a|s,theta)] )

            if use_entropy: entropy += a_probs.entropy().mean()
            reward=r if normalize else torch.tensor([[r]],device=device,dtype=torch.float32)
            log_probs.append(log_a); values.append(v); rewards.append(reward) #Store [to a history buffer] (e.g. via append): log(a), V_critic, r
        
            episode_r += r #increment episode reward
            
            if done: break
        
        #Running Reward
        running_reward=e*episode_r+(1-e)*running_reward #Update running reward to check condition for solving: e=[0.01 ... 0.05]
        if running_reward>=best_rr[0]: 
            best_rr[0]=copy(running_reward)
            best_rr[1]=copy(episode)
            torch.save(model.state_dict(), file_name_own)
        plot_rr.append(running_reward)

        #Consecutive Good Trials
        con_r=con_r+1 if episode_r>=r_thr else 0 #no. of consecutive trials where reward>=r_thr
        max_con=con_r if con_r>max_con else max_con
        
        if learn_std: epsilon.append(np.array(explore_rate).mean())
        
        #Early stopping due reward not improving
        patience=patience+1 if episode_r<=old_er else 0
        old_er=copy(episode_r)
        if patience>=pat_thr:
            print("Early stoppage: Patience ran out!")
            break
        
        #Calculate expected returns (i.e. calculate expected value from rewards) = G or V_pi:
        discounted_sum=0. #assuming non-negative values, since the last env returned done, the value of the last state could be set to 0 #OR:
        # _,v_dash=model(s); discounted_sum=v_dash.clone().cpu().detach().numpy().flatten() #discounted_sum=v_dash
        for R in rewards[::-1]:
            discounted_sum=R+gamma*discounted_sum
            returns.insert(0, discounted_sum) #insert to front of list
        
        #[Optional] Normalize returns (+ don't convert rewards to tensor)
        if normalize:
            returns-=np.mean(returns)
            returns/=(np.std(returns)+np.finfo(np.float32).eps)
            returns=[torch.tensor([[ret]],device=device,dtype=torch.float32) for ret in returns]
        
        #concatenate log_probs, returns & values history lists to appropriate tensor shapes
        log_probs=torch.cat(log_probs)
        returns=torch.cat(returns).detach()
        values=torch.cat(values)
        
        delta=(returns-values).detach() #diff/delta/TD_error/advantage=returns-values
        
        #Calculate loss values (from loss functions) to update our network
        actor_loss=-(log_probs*delta).mean() if avg else -(log_probs*delta).sum() #look also into the categorical loss trick to calculate log(a)
        critic_loss=loss_func(values,returns)
        if asynch:
            critic_loss=vf_coef * critic_loss - ent_coef * entropy if use_entropy else critic_loss
            #Backward propagation:
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)
            if clip_grads: nn.utils.clip_grad_norm_(model.critic.parameters(),max_norm) #nn.utils.clip_grad_norm_(model.actor.parameters(),max_norm) #Clip gradients to avoid exploding grads problem
            optimizer_actor.step() #apply gradients to model trainable params
            optimizer_critic.step()
        else:
            loss=actor_loss + vf_coef * critic_loss - ent_coef * entropy if use_entropy else actor_loss+critic_loss #total loss
            #Backward propagation:
            loss.backward() #Calculate gradients from loss
            if clip_grads: nn.utils.clip_grad_norm_(model.parameters(),max_norm) #Clip gradients to avoid exploding grads problem
            optimizer.step() #apply gradients to model trainable params
        
        # Log & plot info.
        if episode % 10 == 0:
            log_msg="Running Reward: {:.2f} & Episode Reward: {:.2f}".format(running_reward, episode_r)
            t.set_description(desc=log_msg); t.refresh()
        
        # if con_r>=100: or running_reward >= r_thr:  #Condition to consider the task solved
        #     torch.save(model.state_dict(), file_name_own)
        #     print("Solved at episode {}!".format(episode))
        #     break

            
    #Results & Plots
    log1="Best Running Reward is: {:.2f} at episode: {}\n".format(best_rr[0],best_rr[1])
    log2="Greatest number of consecutive trials: {}\n".format(max_con)
    print(log1)
    print(log2)
    if n_updates is None: n_updates=episode if episode<3500 else 3500

    name=folder_op_plt+"train_"+file_name
    f = open(name+".txt", "w")
    f.write(log1)
    f.write(log2)

    title=f"Training Running Reward (Learning Curve) For Own Model: {file_name}"
    plot_func(plot_rr,figsize,title,name,filtered)

    if learn_std:
        name=folder_op_plt+"epsilon_"+file_name
        title=f"Training Exploration Rate (Std of Actor Network Distribution) Average Across Episodes: {file_name}"
        plot_func(plot_rr,figsize,title,name,filtered)


    # ### Validation

    # In[36]:

    print("Own Model Validation ... ")
    model.load_state_dict(torch.load(file_name_own, map_location=device))
    r_val_own_buff=validate(model,env_val,val_eps,file_name,prog,t_limit,asynch)
    log30=f"Number of validation trials : {val_eps} \n"
    log3="Average validation reward for own model is: {} with std={:.2f}\n".format(np.mean(r_val_own_buff),np.std(r_val_own_buff))
    f.write(log30)
    f.write(log3)
    

    # ## Stable Baselines Implementation Benchmark

    # ### Train

    # In[ ]:

    print("SB Model Training ... ")
    model_sb.learn(total_timesteps=n_updates*steps)
    print("SB Model Training Done \n")
    
    # ### Save Model

    # In[29]:

    model_sb.save(file_name_sb)


    # ### Validation

    # In[ ]:


    print("SB Model Validation ... ")
    r_val_sb_buff=validate(model_sb,env_val,val_eps,"sb_"+file_name,prog,t_limit,asynch)

    log4="Average validation reward for SB model is: {} with std={:.2f} \n".format(np.mean(r_val_sb_buff),np.std(r_val_sb_buff))
    f.write(log4)
    f.close()

    return np.mean(r_val_own_buff), np.mean(r_val_sb_buff)
