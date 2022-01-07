import time

import gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import (ClipRewardEnv,
                            EpisodicLifeEnv,
                            FireResetEnv,
                            MaxAndSkipEnv,
                            NoopResetEnv, WarpFrame)
from a2c_ppo_acktr.arguments import get_args


from typing import Callable

class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        if len(env.observation_space.shape) == 3:
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = WarpFrame(env, width=84, height=84)
            env = ClipRewardEnv(env)

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])
        return env
    set_random_seed(seed)
    return _init

def main():
    args = get_args()
    env_id = args.env_name
    # Create the vectorized environment
    env = make_env(env_id)

    model = DQN('CnnPolicy', env, verbose=0)

    # # By default, we use a DummyVecEnv as it is usually faster (cf doc)
    # vec_env = make_vec_env(env_id, n_envs=num_cpu)

    # model = DQN('MlpPolicy', vec_env, verbose=0)

    # We create a separate environment for evaluation
    eval_env = gym.make(env_id)

    # Random Agent, before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

    n_timesteps = 25000

    # Multiprocessed RL Training
    start_time = time.time()
    model.learn(n_timesteps)
    total_time_multi = time.time() - start_time

    print(f"Took {total_time_multi:.2f}s for multiprocessed version - {n_timesteps / total_time_multi:.2f} FPS")

    # Single Process RL Training
    single_process_model = A2C('MlpPolicy', env_id, verbose=0)

    start_time = time.time()
    single_process_model.learn(n_timesteps)
    total_time_single = time.time() - start_time

    print(f"Took {total_time_single:.2f}s for single process version - {n_timesteps / total_time_single:.2f} FPS")

    print("Multiprocessed training is {:.2f}x faster!".format(total_time_single / total_time_multi))

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')


if __name__ == "__main__":
    main()



