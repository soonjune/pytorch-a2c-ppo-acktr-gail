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
from a2c_ppo_acktr.model import Policy, Bandit_Policy
from a2c_ppo_acktr.storage import RolloutStorage, SkipReplayBuffer
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

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, monitor_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    elif args.algo == 'b_a2c':
        nbArms = args.nbArms
        bandit_dim = args.bandit_dim
        bandit = Bandit_Policy(
            envs.observation_space.shape,
            envs.action_space,
            nbArms,
            bandit_dim,
            base_kwargs={'recurrent': args.recurrent_policy})
        bandit.to(device)
        skip_replay_buffer = SkipReplayBuffer(1e7)
        agent = algo.Bandit_A2C_ACKTR(
            actor_critic,
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
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

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

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    log_file = open(log_file_name,'a', newline='')
    log_file_wr = csv.writer(log_file)
    log_file_wr.writerow(['Updates', 'total_num_steps', 'Last 10 mean_episode_rewards'])
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
                    if step+j>0 and len(zero_idx)>0:
                        start_obs[zero_idx] = rollouts.obs[step][zero_idx].clone().detach() # update start skip obs of all processes
                        value[zero_idx], action[zero_idx], action_log_prob[zero_idx], recurrent_hidden_states[zero_idx] = actor_critic.act(
                            rollouts.obs[step][zero_idx], rollouts.recurrent_hidden_states[step][zero_idx],
                            rollouts.masks[step][zero_idx])
                        arms[zero_idx] = bandit.get_skip(rollouts.obs[step][zero_idx], rollouts.recurrent_hidden_states[step][zero_idx],
                        rollouts.masks[step][zero_idx], action[zero_idx], len(zero_idx))
                    else:
                        start_obs = rollouts.obs[step].clone().detach() # reset start skip obs of all processes
                        value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])
                        arms = bandit.get_skip(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step], action, args.num_processes)
                    skips = arms+1
                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])


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
            
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)
            
            # skip replay buffer
            if args.algo == 'b_a2c': 
                skips = masks.squeeze().numpy()*(skips-1)
                zero_idx = np.where(skips==0)[0]
                for idx in zero_idx:
                    skip_replay_buffer.add_transition(start_obs[idx], action[idx], obs[idx], recurrent_hidden_states[idx], reward[idx], masks[idx], arms[idx]) 

        with torch.no_grad():
            next_value = actor_critic.get_value(
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
        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if args.algo == 'b_a2c': 
            # Update Bandit
            batch_skip_obs, batch_skip_actions, batch_skip_next_obs, batch_recurrent_hidden_states, _, \
            batch_masks, batch_skip_arms = skip_replay_buffer.random_next_batch(64)
            # reward 포함
            '''target_rewards = batch_rewards + (1 - batch_terminal_flags) * self._gamma * \
                        torch.max(self._q(batch_next_states), dim=1)[0]'''
            # reward 미포함
            target_rewards = actor_critic.get_value(
                    batch_skip_next_obs, batch_recurrent_hidden_states,
                    batch_masks).detach()

            agent.bandit_train(batch_skip_obs, batch_skip_actions, batch_recurrent_hidden_states, batch_masks,\
                    batch_skip_arms, target_rewards, 64)       

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = save_dir
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
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
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
            log_file = open(log_file_name,'a', newline='')
            log_file_wr = csv.writer(log_file)
            log_file_wr.writerow([j, total_num_steps, np.round(np.mean(episode_rewards),1)])
            log_file.close()

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

if __name__ == "__main__":
    main()
