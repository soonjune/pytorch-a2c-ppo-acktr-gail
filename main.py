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
    

    if args.algo == 'tempo_a2c':
        actor_critic = TempoRLPolicy(
            envs,
            envs.observation_space.shape,
            envs.action_space,
            base_kwargs={'recurrent': args.recurrent_policy},
            skip_dim = args.max_skip_dim)
    else:
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
        agent = algo.Bandit_A2C_ACKTR(
            actor_critic,
            bandit,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo =='tempo_a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
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
    
    if args.algo == 'tempo_a2c':
        skip_rollouts = NoneConcatSkipReplayBuffer(5e4)
        # the skip policy uses epsilon greedy exploration for learning
        initial_expl_noise = args.expl_noise


    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)
    skip_l = []

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    log_file = open(log_file_name,'a', newline='')
    log_file_wr = csv.writer(log_file)
    log_file_wr.writerow(['Updates', 'total_num_steps', 'Last 10 mean_episode_rewards', 'Avg_skips'])
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)
        # implement epsilon decay for tempoRL extension
        if args.algo == 'tempo_a2c':
            expl_noise = initial_expl_noise - (initial_expl_noise * (j / float(num_updates)))

        # t = np.zeros((envs.num_envs,))
        curr_skip = [0 for _ in range(envs.num_envs)]
        skip_graphs = dict()
        for i in range(envs.num_envs): # initialize dict
            skip_graphs[i] = {'skip_states': [], 'skip_rewards': []}# only used for TempoRL to build the local conectedness graph
        repeats = None
        act_continue = None
        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                if args.algo == 'tempo_a2c':
                    if np.random.random() < expl_noise:
                        repeat = np.random.randint(args.max_skip_dim, size=envs.num_envs) # + 1 sonce randint samples from [0, max_rep)
                        # t += repeat
                    else:
                        # print(obs.shape)
                        repeat = np.argmax(actor_critic.get_skip(obs, action), axis=1)
                        # t += repeat
                    if repeats is not None: # previous repeat exists
                        finisehd_repeat = repeats < 0                            
                        repeats = np.array([repeat[idx] if finished else repeats[idx] for idx, finished in enumerate(finisehd_repeat)])
                        act_continue = torch.tensor([action[idx] if finished else act_continue[idx] for idx, finished in enumerate(finisehd_repeat)]).reshape(-1,1).to(device)
                    else:
                        act_continue = action
                        repeats = repeat
                        skip_l.append(np.mean(repeats))
                if args.algo == 'b_a2c':
                    skips = bandit.get_skip(rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], action, args.num_processes)

            # Obser reward and next obs
            if args.algo == 'tempo_a2c':
                prev_obs = obs
                # aug_action = torch.cat((action, torch.from_numpy(repeat.reshape(-1,1)).to(device)), dim=1)
                # Perform action
                obs, reward, done, infos = envs.step(act_continue)
                repeats -= 1
                for i in range(envs.num_envs):
                    skip_graphs[i]['skip_states'].append(prev_obs[i][:])
                    skip_graphs[i]['skip_rewards'].append(reward[i][:])

                # print(obs.shape)
                # print(reward, reward.shape)
                # Update the skip replay buffer with all observed skips.
                for k in skip_graphs.keys():
                    skip_id = 0
                    for start_state in skip_graphs[k]['skip_states']:
                        skip_reward = 0
                        for exp, r in enumerate(skip_graphs[k]['skip_rewards'][skip_id:]):
                            skip_reward += np.power(args.gamma, exp) * r
                        skip_rollouts.add_transition(start_state.cpu().numpy(), curr_skip[k] - skip_id, obs[k].cpu().numpy(), 
                                                     skip_reward.cpu().numpy(), done[i], curr_skip[i] - skip_id + 1,
                                                     act_continue.cpu().numpy()[k])
                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                    for info in infos])
                # Insert repeated rollout trajectories
                rollouts.insert(obs, recurrent_hidden_states, act_continue,
                                action_log_prob, value, reward, masks, bad_masks)
                # Update the skip buffer with all observed transitions in the local connectedness graph
      

            else:
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

        if args.algo == 'tempo_a2c':
            # Skip Q update based on double DQN where target is behavior Q
            batch_states, batch_actions, batch_next_states, batch_rewards,\
                batch_terminal_flags, batch_lengths, batch_behaviours = \
                skip_rollouts.random_next_batch(args.tempo_batch_size)
            target = batch_rewards.squeeze() + (1 - batch_terminal_flags) * torch.pow(batch_lengths, args.gamma)  * \
                        actor_critic.get_value(batch_next_states, False, None)[torch.arange(args.tempo_batch_size).long(), torch.argmax(
                            actor_critic.get_value(batch_next_states, False, None), dim=1)]
            current_prediction = actor_critic.skip_Q(batch_states, batch_behaviours)[
                        torch.arange(args.tempo_batch_size).long(), batch_actions.long()]
            
            loss = actor_critic.skip_loss_function(current_prediction, target.detach())
            loss.backward()
            actor_critic.skip_optimizer.step()

    
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
            log_file_wr.writerow([j, total_num_steps, np.round(np.mean(episode_rewards), 1), np.round(np.mean(skip_l), 1)])
            log_file.close()

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            obs_rms = utils.get_vec_normalize(envs).obs_rms
            evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                     args.num_processes, eval_log_dir, device)

if __name__ == "__main__":
    main()
