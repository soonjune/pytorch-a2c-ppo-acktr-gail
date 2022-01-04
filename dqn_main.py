import gym

from stable_baselines3 import DQN

import copy
import glob
import os
import time
import datetime
import csv
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy, TempoRLPolicy, Bandit_Policy
from a2c_ppo_acktr.storage import RolloutStorage, NoneConcatSkipReplayBuffer
from evaluation import evaluate

def main():
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    time_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S.%f')
    monitor_dir = os.path.expanduser(args.log_dir) + '/' + args.env_name + '/' + args.algo

    log_dir = monitor_dir + '/' + time_str 
    save_dir = monitor_dir + '/' + time_str + '/' + args.save_dir
    eval_log_dir = monitor_dir + '/' + time_str + '/' + 'eval'
    log_file_name = os.path.join(log_dir, "log.csv")

    utils.cleanup_log_dir(monitor_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    if args.algo == 'tempo_a2c':
        action_repeat = True
    else:
        action_repeat = False
    

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, monitor_dir, device, False, None, action_repeat)

    model = DQN("CNNPolicy", envs, verbose=1)
    model.learn(total_timesteps=int(2e5))
    model.save("dqn_breakout")
    # del model  # delete trained model to demonstrate loading    


    obs = envs.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = envs.step(action)
        envs.render()
        if done:
            obs = envs.reset()
        

        

