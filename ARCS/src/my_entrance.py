import os
os.environ["OMP_NUM_THREADS"] = "1"         
os.environ["MKL_NUM_THREADS"] = "1"        
os.environ["TF_NUM_INTRAOP_THREADS"] = "1" 
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import gym
import gym_compete

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import logger as sb_logger

if not hasattr(gym.spaces.Box, 'dtype'):
    @property
    def dtype(self):
        return self.low.dtype
    gym.spaces.Box.dtype = dtype

from scheduling import ConstantAnnealer, Scheduler
from shaping_wrappers import apply_reward_wrapper
from environment_my import make_zoo_multi2single_env
from mask_env import make_mixadv_multi2single_env
from RewardDic import Reward_Dic
from my_PPO import ParallelPPOTrainer
import argparse

ENV_LIST = ['multicomp/SumoHumans-v0', 'multicomp/YouShallNotPassHumans-v0', 'multicomp/KickAndDefend-v0']
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=88, help="random seed")
parser.add_argument("--mode", type=str, choices=["abs", "oppo", "llm", "mask", "retrain_abs", "retrain_oppo" ,"retrain_llm"], default="llm", help="reward mode")
parser.add_argument("--env", type=str, default='multicomp/SumoHumans-v0', help="environment name")
parser.add_argument("--steps", type=float, default=3.5e7, help="max training steps")
parser.add_argument("--num_envs", type=int, default=8, help="number of parallel envs")
parser.add_argument('--use_entropy', action='store_true', help="use_entropy or not")
args = parser.parse_args()
seed = args.seed
mode = args.mode
env_name = args.env
num_envs = args.num_envs
max_train_steps = int(args.steps)
use_entropy = args.use_entropy

if env_name == 'multicomp/SumoHumans-v0':
    idv = 3
else:
    idv = 1

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

REW_SHAPE_PARAMS = {
    'weights': {
        'dense': {'reward_move': 0.1},
        'sparse': {'reward_remaining': 0.01}
    },
    'anneal_frac': 0    
}
scheduler = Scheduler(annealer_dict={'lr': ConstantAnnealer(3e-4)})

if mode == "mask":
    adv_agent_norm_path = "/path/to/VecNormalize_weights.pth"
    adv_agent_path = "/path/to/adv_agent_weights.pth"

if 'You' in env_name.split('/')[1]:
    REVERSE = True
else:
    REVERSE = False

if "llm" in mode:
    reward_str = Reward_Dic[env_name]
else:
    reward_str = None

if "mask" in mode:
    env = SubprocVecEnv([
        (lambda idx=i: make_mixadv_multi2single_env(
            env_name, idv, adv_agent_path, adv_agent_norm_path, REW_SHAPE_PARAMS, scheduler, max_train_steps, n_envs=1, reverse=REVERSE, mode=mode, seed=seed+idx, reward_str=reward_str
        ))  for i in range(num_envs)
    ])
    env = apply_reward_wrapper(single_env=env, scheduler=scheduler,
                                        agent_idx=0, shaping_params=REW_SHAPE_PARAMS,
                                        total_step=2e7)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env_test = DummyVecEnv([lambda: make_mixadv_multi2single_env(
        env_name, idv, adv_agent_path, adv_agent_norm_path, REW_SHAPE_PARAMS, scheduler, max_train_steps, n_envs=1, reverse=REVERSE, mode=mode,seed=seed+100, reward_str=reward_str
        )])
    env_test = VecNormalize(
                env_test,
                norm_obs=True,
                norm_reward=False, 
                training=False
            )

else:
    env = SubprocVecEnv([
            (lambda idx=i: make_zoo_multi2single_env(
                env_name, idv, REW_SHAPE_PARAMS, scheduler,
                reverse=REVERSE, total_step=max_train_steps, seed=seed+idx,
                mode=mode, reward_str=reward_str
            ))  for i in range(num_envs)
        ])

    if "abs" in mode or "oppo" in mode:
        env = apply_reward_wrapper(single_env=env, scheduler=scheduler,
                                            agent_idx=0, shaping_params=REW_SHAPE_PARAMS,
                                            total_step=max_train_steps)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env_test = DummyVecEnv([
        lambda: make_zoo_multi2single_env(
            env_name, idv, REW_SHAPE_PARAMS, scheduler,
            reverse=REVERSE, total_step=max_train_steps, seed=seed+100,
            mode=mode, reward_str=reward_str
        )
    ])
    env_test = VecNormalize(
        env_test,
        norm_obs=True,
        norm_reward=False, 
        training=False
    )

trainer = ParallelPPOTrainer(
    env,
    env_test,
    num_envs=num_envs,
    max_train_steps=max_train_steps,
    seed=seed,
    mode=mode,
    use_entropy=use_entropy,
    reward_str=reward_str,
    device="cpu"
)
trainer.train()