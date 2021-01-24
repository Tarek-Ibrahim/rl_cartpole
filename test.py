#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


#Essentials
import numpy as np
import matplotlib.pyplot as plt

#Funcs
from utils import test, plot_func
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

#Stable-baselines3
from stable_baselines3 import A2C
from stable_baselines3.a2c import MlpPolicy
from stable_baselines3.common.env_checker import check_env


def testing(env_id,gamma,test_episodes,best_model,best_model_sb,PID,vis,figsize,trim,t_limit,action_std,
filtered,enforce_smooth,normalize,learn_std,asynch):

    # # Initializations

    # In[5]:

    #Device
    use_cuda=torch.cuda.is_available()
    device=torch.device("cuda" if use_cuda else "cpu")

    #Best Model
    h,lr,env_id_own=best_model
    h_sb,lr_sb,env_id_sb=best_model_sb

    #Environment
    env_te=gym.make("env_te-v"+str(env_id))
    env_te.env.enforce_smooth=enforce_smooth
    env_te_pid=gym.make("env_te-v"+str(env_id))
    env_te_pid.env.enforce_smooth=False
    n=env_te.observation_space.shape[0] #length of state vector (=4) #state is continous #[x,v,theta,omega] #velocities=|R #x=+-4.8 #theta=+-24 degrees
    m=env_te.action_space.shape[0] #no. of actions #continous actions [-1,1]

    #Hyperparameters
    #method
    astr="1e"+str(decimal.Decimal(str(lr)).as_tuple().exponent)
    astr_sb="1e"+str(decimal.Decimal(str(lr_sb)).as_tuple().exponent)
    #Model
    model=ActorCritic(input_size=n,output_size=m,hidden_size_1=h,hidden_size_2=h,device=device,action_std=action_std,
    normalize=normalize,learn_std=learn_std).to(device) #Own Model Init

    #Misc
    prog=lambda x: tqdm.trange(x, leave=True)
    T=100 if trim is None else trim
    folder_own="models/own/"
    folder_sb="models/sb/"
    folder_op_plt="output/results_and_plots/test/"
    folder_tmp="output/tmp/"
    folder_op_vid="output/video_demos/"
    env_name_own=f"ac_env{env_id}_"
    env_name_sb=f"ac_env{env_id}_"
    env_name_pid=f"env{env_id}"
    model_name=f"h{h}_alpha{astr}"
    model_name_sb=f"h{h_sb}_alpha{astr_sb}"
    file_name_own=env_name_own+model_name
    file_name_sb=env_name_sb+model_name_sb
    file_name_pid=env_name_pid
    loaded_model_own=folder_own+f"ac_env{env_id_own}_"+model_name+".pt"
    loaded_model_sb=folder_sb+f"ac_env{env_id_sb}_"+model_name


    # # Implementation

    # ## Our Implementation

    # ### Load Model

    # In[4]:

    model.load_state_dict(torch.load(loaded_model_own, map_location=device))


    # ### Test

    # In[ ]:

    print("Own Model Testing ... ")
    name_vid=folder_op_vid +"own_"+file_name_own
    r_te_own_buff, cas, states =test(model,env_te,test_episodes,file_name_own,name_vid,folder_tmp,prog,vis,t_limit,asynch)


    # ### Control actions sample

    # In[27]:

    filtered_cas="poly"

    for idx,ca in enumerate(cas):
        if len(ca)>=T:
            break

    ca_plot=copy(ca) if trim is None else copy(ca[:trim])
    title=f"Force Magnitude (Control Action) Values for Testing of Own Model {file_name_own} at Episode {idx}"
    name=folder_op_plt+"control_actions/"+"own_ca_"+file_name_own
    plot_func(ca_plot,figsize,title,name,filtered_cas)

    with open(name+".txt","w") as f:
        for item in ca:
            f.write(f"{item} ")
    
    f.close()

    states_plot=copy(states[idx]) if trim is None else copy(states[idx][:trim])
    title=f"States evolution for Testing of Own Model {file_name_own} at Episode {idx}"
    name=folder_op_plt+"states/"+"own_states_"+file_name_own
    labels=('x','x_dot','theta','theta_dot')
    plot_func(states_plot,figsize,title,name,filtered_cas,labels=labels)

    with open(name+".txt","w") as f:
        for item in states[idx]:
            f.write(f"{item} ")
    
    f.close()


    # ## Stable Baselines Implementation Benchmark

    # ### Load Model

    # In[45]:

    model_sb = A2C.load(loaded_model_sb,device=device)


    # ### Test

    # In[38]:

    print("SB Model Testing ... ")
    name_vid=folder_op_vid +"sb_"+file_name_sb
    r_te_sb_buff, cas_sb, states_sb =test(model_sb,env_te,test_episodes,file_name_sb,name_vid,folder_tmp,prog,vis,t_limit,asynch,model_type="sb")

    # ### Control actions sample

    # In[41]:

    for idx,ca in enumerate(cas_sb):
        if len(ca)>=T:
            break
    ca_plot=copy(ca) if trim is None else copy(ca[:trim])
    title=f"Force Magnitude (Control Action) Values for Testing of SB Model {file_name_sb} at Episode {idx}"
    name=folder_op_plt+"control_actions/"+"sb_ca_"+file_name_sb
    plot_func(ca_plot,figsize,title,name,filtered_cas)

    with open(name+".txt","w") as f:
        for item in ca:
            f.write(f"{item} ")
    
    f.close()

    states_plot=copy(states_sb[idx]) if trim is None else copy(states_sb[idx][:trim])
    title=f"States evolution for Testing of SB Model {file_name_sb} at Episode {idx}"
    name=folder_op_plt+"states/"+"sb_states_"+file_name_sb
    plot_func(states_plot,figsize,title,name,filtered_cas,labels=labels)

    with open(name+".txt","w") as f:
        for item in states_sb[idx]:
            f.write(f"{item} ")
    
    f.close()

    # ## Random Policy Implementation

    # In[18]:


    print("Random Model Testing ... ")
    name_vid=folder_op_vid +"rand"+file_name_pid
    reward_rand_buff, _ , _=test(None,env_te,test_episodes,"rand_"+file_name_pid,name_vid,folder_tmp,prog,vis,t_limit,asynch,model_type="rand")


    # ## PID

    print("PID Model Testing ... ")
    name_vid=folder_op_vid +"pid_"+file_name_pid
    r_te_pid_buff, cas_pid, states_pid =test(PID,env_te_pid,test_episodes,"pid_"+file_name_pid,name_vid,folder_tmp,prog,vis,t_limit,asynch,model_type="pid")

    # ### Control actions sample

    # In[41]:

    for idx,ca in enumerate(cas_pid):
        if len(ca)>=T:
            break
    ca_plot=copy(ca) if trim is None else copy(ca[:trim])
    title=f"Force Magnitude (Control Action) Values for Testing of PID Model {file_name_pid} at Episode {idx}"
    name=folder_op_plt+"control_actions/"+"pid_ca_"+file_name_pid
    plot_func(ca_plot,figsize,title,name,filtered)

    with open(name+".txt","w") as f:
        for item in ca:
            f.write(f"{item} ")
    
    f.close()

    states_plot=copy(states_pid[idx]) if trim is None else copy(states_pid[idx][:trim])
    title=f"States evolution for Testing of PID Model {file_name_pid} at Episode {idx}"
    name=folder_op_plt+"states/"+"pid_states_"+file_name_pid
    plot_func(states_plot,figsize,title,name,filtered,labels=labels)

    with open(name+".txt","w") as f:
        for item in states_pid[idx]:
            f.write(f"{item} ")
    
    f.close()


    # # Results

    # In[10]:


    #Average reward
    logall1="No. of testing trials: {} \n".format(test_episodes)
    logall2="Average reward for own model is: {} with std={:.2f} \n".format(np.mean(r_te_own_buff),np.std(r_te_own_buff))
    logall3="Average reward for sb model is: {} with std={:.2f} \n".format(np.mean(r_te_sb_buff),np.std(r_te_sb_buff))
    logall4="Average reward for random model is: {} with std={:.2f} \n".format(np.mean(reward_rand_buff),np.std(reward_rand_buff))
    logall5="Average reward for PID model is: {} with std={:.2f} \n".format(np.mean(r_te_pid_buff),np.std(r_te_pid_buff))
    print(logall1)
    print(f"Own Model : {file_name_own} \n")
    print(f"SB Model : {file_name_sb} \n")
    print(logall2)
    print(logall3)
    print(logall4)
    print(logall5)

    #Plots
    plt.figure(figsize=figsize)
    plt.grid(1)

    plt.plot(mov_avg(r_te_own_buff, size=3,mode='nearest'),label="own rewards")
    plt.plot(mov_avg(r_te_sb_buff, size=3,mode='nearest'),label="sb rewards")
    plt.plot(mov_avg(reward_rand_buff, size=3,mode='nearest'),label="rand rewards")
    plt.plot(mov_avg(r_te_pid_buff, size=3,mode='nearest'),label="pid rewards")

    plt.axhline(y=np.mean(r_te_own_buff),color='b',label="own avg reward")
    plt.axhline(y=np.mean(r_te_sb_buff),color='r',label="sb avg reward")
    plt.axhline(y=np.mean(reward_rand_buff),label="rand avg reward")
    plt.axhline(y=np.mean(r_te_pid_buff),label="pid avg reward")

    title=f"Overall Testing Results for testing env={env_id}"
    plt.title(title)
    plt.legend()

    name=folder_op_plt+f"overall_best_env{env_id}"

    f = open(name+".txt", "w")
    f.write(logall1)
    f.write(logall2)
    f.write(logall3)
    f.write(logall4)
    f.write(logall5)
    f.close()

    plt.savefig(name+'.png')

    # plt.show()

