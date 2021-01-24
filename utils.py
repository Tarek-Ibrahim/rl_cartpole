#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


#Essentials
import numpy as np
import matplotlib.pyplot as plt

#Utils
import os
import glob
import tqdm
from time import time
from copy import copy, deepcopy
from scipy.ndimage import uniform_filter1d as mov_avg
from scipy.signal import savgol_filter
from moviepy.editor import concatenate_videoclips, VideoFileClip
from scipy.interpolate import make_interp_spline, BSpline

#Envs.
import gym
from gym import wrappers

#ML Framework (PyTorch)
import torch
import torch.nn as nn


# # Functions

# In[2]:


def validate(model,env,val_eps,model_name,prog,t_limit,asynch):
    
    reward_buffer=[]
    t=prog(val_eps)

    for _ in t:

        episode_r=0. #Init episode reward
        s = env.reset()
        start=time()

        while True:
            if "sb" in model_name:
                a,_=model.predict(s)
            else:
                if asynch:
                    dist=model.actor_net(s)
                    a=dist.sample().cpu().numpy().flatten()
                else:
                    dist, _=model(s)
                    a=dist.sample().cpu().numpy()[0]
            s, r, done, _ = env.step(a)
            episode_r += r

            if done: break
            end=time()
            elapsed=end-start
            if elapsed>t_limit:
                _, _, done, _=env.step([1e6])
                break

        reward_buffer.append(episode_r)
        log_msg="Episode Reward: {:.2f}".format(episode_r)
        t.set_description(desc=log_msg); t.refresh()

    print("Average validation reward for model {} is: {} with std={:.2f} \n".format(model_name,np.mean(reward_buffer),np.std(reward_buffer)))
    return reward_buffer


def hyperopt(avgrs,avgrs_sb,figsize,env_ids):

    max_avgr=max(list(avgrs.keys()))
    max_avgr_sb=max(list(avgrs_sb.keys()))
    best_model=avgrs[max_avgr]
    best_model_sb=avgrs_sb[max_avgr_sb]
    h,lr,env_id_own=best_model
    h_sb,lr_sb,env_id_sb=best_model_sb
    print(f"Best own model is on env={env_id_own} and with h={h} & alpha={lr} \n")
    print(f"Best SB model is on env={env_id_sb} and with h={h_sb} & alpha={lr_sb} \n")

    # Hyperparameter Exploration Plot
    pltname="Hyperparameter_Space_Exploration"
    plt.figure(figsize=(8,8))
    plt.grid(1)
    plt.xlabel("env_id")
    plt.ylabel("Avg Testing Reward")
    for key in avgrs.keys():
        h,lr,env_id=avgrs[key]
        plt.scatter(env_id,key,color="b")
        plt.text(env_id, key, "own_"+str(env_id)+f"_h{h}_lr{lr}")
    for key_sb in avgrs_sb.keys():
        h,lr,env_id=avgrs_sb[key_sb]
        plt.scatter(env_id,key_sb,color="r")
        plt.text(env_id, key_sb, "sb_"+str(env_id)+f"_h{h}_lr{lr}",horizontalalignment="right")
    plt.xticks(env_ids)
    plt.title(pltname)
    plt.savefig("output/results_and_plots/"+pltname+'.png')
    # plt.show()

    return best_model, best_model_sb


def test(model,env_te,te_eps,model_name,vid_name,tmp,prog,vis,t_limit,asynch,model_type="own"):
    
    env = wrappers.Monitor(env_te,tmp,force=True,resume=False, write_upon_reset=False,video_callable=lambda episode_id: episode_id%10==0)
    reward_buffer=[]
    control_actions=[]
    states=[]
    t=prog(te_eps)

    #PID inits.
    if model_type=="pid": P,I,D=model
    desired_state = np.array([0, 0, 0, 0])
    desired_mask = np.array([1, 0, 1, 1])

    for _ in t:

        control_action=[]
        state=[]
        episode_r = 0.
        s = env.reset()
        if vis: env.render()
        start=time()

        #PID params
        integral = 0
        derivative = 0
        prev_error = 0

        while True:

            #Determine action
            if model_type=="rand":
                a=env.action_space.sample()
            elif model_type=="sb":
                a,_=model.predict(s)
            elif model_type=="pid":
                s[2]*=180./np.pi
                error=s-desired_state
                integral += error
                derivative = error - prev_error
                prev_error = error
                pid = np.dot(P * error + I * integral + D * derivative, desired_mask).sum()
                a=[np.tanh(pid)]
            else:
                if asynch:
                    dist=model.actor_net(s)
                    a=dist.sample().cpu().numpy().flatten()
                else:
                    dist, _=model(s)
                    a=dist.sample().cpu().numpy()[0]

            control_action.append(env.env.force)
            state.append(s)
            s, r, done, _ = env.step(a)
            episode_r += r
            if vis: env.render()
            if done: break
            #Time-box each episode: if episode length exceeds "t_limit" seconds, assign the maximal reward to the episode (consider it as endless) and move on to the next one
            end=time()
            elapsed=end-start
            if elapsed>t_limit:
                _, _, done, _=env.step([1e6])
                break

        reward_buffer.append(episode_r)
        control_actions.append(control_action)
        states.append(state)

        log_msg="Episode Reward: {:.2f}".format(episode_r)
        t.set_description(desc=log_msg); t.refresh()

    env.close()
    env_te.close()
    
    L =[]
    for root, _, files in os.walk(tmp):
        for file in files:
            if os.path.splitext(file)[-1] == '.mp4':
                filePath = os.path.join(root, file)
                video = VideoFileClip(filePath)
                L.append(video)
        
    final_clip = concatenate_videoclips(L)
    final_clip.write_videofile(vid_name+".mp4", fps=24, remove_temp=False,verbose=False,logger=None)
    
    final_clip.close()
    for vid in L: vid.close()
    
    files = glob.glob(tmp+'*')
    for f in files: os.remove(f)
    
    print("Average testing reward for model {} is: {} with std={:.2f}\n".format(model_name,np.mean(reward_buffer),np.std(reward_buffer)))
    return reward_buffer, control_actions, states

    
def plot_func(val,figsize,title,name,filtered,labels=None):
    plt.figure(figsize=figsize)
    plt.grid(1)
    if filtered is None:
        plt.plot(val)
    elif filtered =="mov_avg":
        plt.plot(mov_avg(val, size=3,mode='nearest'))
    elif filtered =="savgol":
        plt.plot(savgol_filter(val, 5, 3)) # window size, polynomial order
    elif filtered=="poly":
        xm=range(len(val))
        x=np.linspace(min(xm),max(xm),len(val)*6)
        spl = make_interp_spline(xm, val, k=3)
        valnew = spl(x)
        # plt.plot(val)
        plt.plot(x, valnew)
    plt.title(title)
    if labels is not None: plt.legend(labels)
    plt.savefig(name+'.png')
    # plt.show()


def args_help(s_opts,l_opts):
    print("Usage: \n")
    print("python main.py [-<short_flag>=<value>] [--long_flag=<value>] \n")
    print("<value> depends on the variable's DT and could be numeric, boolean, None, list or tuple \n")
    print("Argumensts: (Note: for arguments meaning and default value check main.py documentation) \n")
    for idx,i in enumerate(s_opts):
        if i != ":" and s_opts[idx+1]==":":
            print("-"+i+" , "+"--"+l_opts[int(np.ceil(idx/2))][:-1]+"\n")


def plot_ca_states(trim,figsize):

    labels=('x','x_dot','theta','theta_dot')
    states_title= lambda name: f"States evolution for Testing of Model {name} at Episode 0"
    ca_title=lambda name: f"Force Magnitude (Control Action) Values for Testing of Model {name} at Episode 0"
    titles=[ca_title,states_title]

    ca_folder="output/results_and_plots/test/control_actions/"
    states_folder="output/results_and_plots/test/states/"
    folders=[ca_folder,states_folder]

    for idx,folder in enumerate(folders):
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[-1] == '.txt':
                    filename=os.path.splitext(f)[:-1][0]
                    filePath = os.path.join(root, f)
                    with open(filePath,"r") as fh:
                        for line in fh:
                            x=line.split(" ")
                    if idx==1:
                        val=[]
                        for i in x:
                            if len(i)<2:
                                continue
                            elif i[0]=="[":
                                val.append(i[1:])
                            elif i[-1]=="]":
                                val.append(i[:-1])
                            else:
                                val.append(i)
                        val=[[val[i],val[i+1],val[i+2],val[i+3]] for i in range(0,len(val)-3,4)]
                    else:
                        val=x[:-1]
                    val=val[:trim] if trim is not None else val[:]
                    fh.close()
                    plt.figure(figsize=figsize)
                    plt.grid(1)
                    xm=range(len(val))
                    x=np.linspace(min(xm),max(xm),len(val)*6)
                    spl = make_interp_spline(xm, val, k=3)
                    valnew = spl(x)
                    # plt.plot(val)
                    plt.plot(x, valnew)
                    title=titles[idx](filename)
                    plt.title(title)
                    if folder==states_folder: plt.legend(labels)
                    plt.savefig(os.path.join(root, filename)+'.png')
