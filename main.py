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
from a2c_ppo_acktr.model import Policy, Bandit_Policy, Perturb_Bandit_Policy
from a2c_ppo_acktr.algo.bandit_dqn import NNLinTS
from a2c_ppo_acktr.storage import RolloutStorage, SkipReplayBuffer
from evaluation import evaluate

from sb3_contrib import QRDQN, TQC, TRPO
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3


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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, monitor_dir, device, False)

    if args.pre_trained:
        num = 'a2c' + '_3'
        path = f"./experiments/{args.env_name}/a2c/{num}/trained_models/{args.env_name}.pt"
        model = torch.load(path)[0]
    else:
        model = Policy(
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy})
    model.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            model,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppo':
        agent = algo.PPO(
            model,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    elif args.algo in ['b_a2c','b_ppo'] :
        nbArms = args.nbArms
        bandit_dim = args.bandit_dim
        bandit = Bandit_Policy(
            envs.observation_space.shape,
            envs.action_space,
            nbArms,
            bandit_dim,
            base_kwargs={'recurrent': args.recurrent_policy})
        bandit.to(device)
        skip_replay_buffer = SkipReplayBuffer(1000)
        agent = algo.Bandit_A2C_ACKTR(
            model,
            bandit,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
        skips = np.zeros(args.num_processes)
        zero_idx = np.arange(args.num_processes)

    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            model, args.value_loss_coef, args.entropy_coef, acktr=True)

    elif args.algo in ['dqn', 'bdqn']:
        assert args.num_processes ==1 and args.num_steps ==1
        pretrained_agent = 'dqn'
        model_path = os.path.join('./rl-trained-agents/',pretrained_agent, args.env_name+"_1", args.env_name+".zip")
        model = DQN.load(model_path, env=envs)
        deterministic = False
        state = None
        if args.algo.startswith('b'):
            nbArms = args.nbArms
            bandit_dim = args.bandit_dim
            bandit = NNLinTS(
            model,
            3136,
            envs.action_space.n,
            nbArms,
            bandit_dim)
            bandit.to(device)
            #skip_replay_buffer = SkipReplayBuffer(80)  # updates with collected samples
            skip_replay_buffer = SkipReplayBuffer(5e4)  

            skips = np.zeros(args.num_processes)
            zero_idx = np.arange(args.num_processes)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)

    if  not(args.algo.endswith('dqn')):
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                envs.observation_space.shape, envs.action_space,
                                model.recurrent_hidden_state_size)
        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)
    else:
        obs = envs.reset()

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    log_file = open(log_file_name,'a', newline='')
    log_file_wr = csv.writer(log_file)
    log_file_wr.writerow(['Updates', 'total_num_steps', 'Last 10 mean_episode_rewards', 'Avg_skips'])

    skip_update_count = 0
    skips_l = []
    prev_action = np.zeros(args.num_processes)-999
    virtual_arm = np.zeros(args.num_processes)
    skip_obs = [obs]

    for j in range(num_updates):
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.algo == 'b_a2c':
                    if step+j>0:
                        if len(zero_idx)>0: # skip update
                            start_obs[zero_idx] = rollouts.obs[step][zero_idx].clone().detach() # update start skip obs 
                            value[zero_idx], action[zero_idx], action_log_prob[zero_idx], recurrent_hidden_states[zero_idx] = model.act(
                                rollouts.obs[step][zero_idx], rollouts.recurrent_hidden_states[step][zero_idx],
                                rollouts.masks[step][zero_idx])
                            arms[zero_idx] = bandit.get_skip(rollouts.obs[step][zero_idx], rollouts.recurrent_hidden_states[step][zero_idx],
                            rollouts.masks[step][zero_idx], action[zero_idx], len(zero_idx))  
                            skips[zero_idx] = arms[zero_idx] + 1                    
                    else: # initialization
                        start_obs = rollouts.obs[step].clone().detach() # reset start skip obs of all processes
                        value, action, action_log_prob, recurrent_hidden_states = model.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                        arms = bandit.get_skip(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step], action, args.num_processes)
                        skips = arms+1
                elif args.algo in ['dqn']:
                    action, _ = model.predict(obs.numpy(), state=state, deterministic=deterministic)
                    action = torch.tensor(np.array([action]))
                    skips = 1
                elif args.algo in ['bdqn']:
                    if skips[0] == 0:
                        start_obs = obs.clone().detach() # reset start skip obs of all processes
                        action, _ = model.predict(obs.numpy(), state=state, deterministic=deterministic)
                        action = torch.tensor(np.array([action]))
                        arms = bandit.get_skip(start_obs, action, args.num_processes)                
                        skips = arms+1          
                else:
                    value, action, action_log_prob, recurrent_hidden_states = model.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                    skips = 1

            skips_l.append(np.mean(skips))
            
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            
            if not(args.algo.endswith('dqn')):
                rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
            # add skip history to skip replay buffer 
            if args.algo.startswith('b'):  
                skips = masks.squeeze().numpy()*(skips-1)
                '''zero_idx = np.where(skips==0)[0]
                for idx in zero_idx:
                    skip_update_count+=1
                    if not(args.algo.endswith('dqn')):
                        skip_replay_buffer.add_transition(start_obs[idx], action[idx], obs[idx],\
                            recurrent_hidden_states[idx], reward[idx], masks[idx], arms[idx]) 
                    else:
                        skip_replay_buffer.add_transition(start_obs[idx], action[idx], obs[idx],\
                            torch.zeros(1,1)[idx], reward[idx], masks[idx], arms[idx]) '''

                # update with virtual extension ######################################  
                if (prev_action[0]==action[0]) and (masks.squeeze().numpy()!=0):
                    virtual_arm = virtual_arm+1
                    if len(skip_obs) > (args.nbArms):
                        skip_obs.pop(0)      
                else:
                    virtual_arm = np.zeros(args.num_processes)
                    skip_obs=[skip_obs[-1]]
                skip_obs.append(obs)
                #print(masks.squeeze().numpy(), action[0],virtual_arm[0], len(skip_obs))
                skip_update_count+=1
                skip_replay_buffer.add_transition(skip_obs[0][0], action[0], skip_obs[-1][0],\
                            torch.zeros(1,1)[0], reward[0], masks[0], virtual_arm[0]) 
                
            prev_action = copy.deepcopy(action)
            ###########################################################################
            
            # update bandit
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            if (args.algo == 'b_a2c') and (skip_update_count==args.num_mini_batch): 
                # Update Bandit
                batch_skip_obs, batch_skip_actions, batch_skip_next_obs, batch_recurrent_hidden_states, _, \
                batch_masks, batch_skip_arms = skip_replay_buffer.recent_batch_sample(skip_update_count)
                # reward 포함
                '''target_rewards = batch_rewards + batch_masks.detach().to(device) * self._gamma * \
                            torch.max(self._q(batch_next_states), dim=1)[0]'''
                # reward 미포함
                target_rewards = batch_masks.detach().to(device)*model.get_value(    # test 1.batch_masks 추가
                        batch_skip_next_obs, batch_recurrent_hidden_states,
                        batch_masks).detach()

                agent.bandit_train(batch_skip_obs, batch_skip_actions, batch_recurrent_hidden_states, batch_masks,\
                        batch_skip_arms, target_rewards, skip_update_count)  
                skip_update_count = 0   
                skip_replay_buffer.reset()

            # random sample
            elif (args.algo == 'bdqn') :
                if (j % args.bandit_update_interval == 0) and (skip_replay_buffer._size>args.num_mini_batch):     
                    # update banditnet           
                    batch_skip_obs, batch_skip_actions, batch_skip_next_obs, _, _, \
                    batch_masks, batch_skip_arms = skip_replay_buffer.random_next_batch(args.num_mini_batch)
                    target_rewards = batch_masks.detach().to(device)*torch.max(model.policy.q_net(obs.to(device).float()), dim=1)[0].detach()
                    bandit.update_banditnet(batch_skip_obs, batch_skip_actions, batch_skip_arms, target_rewards, args.num_mini_batch)
                if (skip_replay_buffer._size>0) and (skip_update_count == args.num_mini_batch):
                    # update TS
                    batch_skip_obs, batch_skip_actions, batch_skip_next_obs, _, _, \
                    batch_masks, batch_skip_arms = skip_replay_buffer.recent_batch_sample(args.num_mini_batch)
                    target_rewards = batch_masks.detach().to(device)*torch.max(model.policy.q_net(obs.to(device).float()), dim=1)[0].detach()
                    bandit.update_TS(batch_skip_obs, batch_skip_actions, batch_skip_arms, target_rewards, args.num_mini_batch)
                    skip_update_count = 0

            '''# updates with collected samples
            elif (args.algo == 'bdqn') and (skip_replay_buffer._size>0) and (skip_replay_buffer._size % args.num_mini_batch == 0) : 
                batch_skip_obs, batch_skip_actions, batch_skip_next_obs, _, _, \
                batch_masks, batch_skip_arms = skip_replay_buffer.recent_batch_sample(args.num_mini_batch)
                target_rewards = batch_masks.detach().to(device)*torch.max(model.policy.q_net(obs.to(device).float()), dim=1)[0].detach()
                bandit.update(batch_skip_obs, batch_skip_actions, batch_skip_arms, target_rewards, args.num_mini_batch)
                skip_replay_buffer.reset()'''
        
        if (args.pre_trained == False) and not(args.algo.endswith('dqn')):
            with torch.no_grad():
                next_value = model.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        # Update Policy
        if (args.pre_trained == False) and not(args.algo.endswith('dqn')):
            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                    args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()  

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = save_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass
            if args.algo in ['bdqn']:
                model.save(save_path)
                torch.save([
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None),
                    bandit.Bandit_Net,
                    bandit.thetaLS,
                ], os.path.join(save_path, args.env_name + ".pt"))
            elif args.algo in ['dqn']:
                model.save(save_path)
                torch.save([
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None),
                ], os.path.join(save_path, args.env_name + ".pt"))
            else:
                torch.save([
                    model,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"\
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards)))
            log_file = open(log_file_name,'a', newline='')
            log_file_wr = csv.writer(log_file)
            log_file_wr.writerow([j, total_num_steps, np.round(np.mean(episode_rewards),1), np.round(np.mean(skips_l),1)])
            log_file.close()
            skips_l = []

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(model, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

if __name__ == "__main__":
    main()
