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
from reward_generator import Reward_Generator
import pickle
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
args = parser.parse_args()
seed = args.seed

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


mode = "llm"
env_name = "multicomp/SumoHumans-v0"

if 'You' in env_name.split('/')[1]:
    REVERSE = True
else:
    REVERSE = False

num_envs = 8
idv = 3 

RewardGenerator = Reward_Generator("llm")
if  os.path.isfile("/home/data/sdb5/jiangjunyong/ARCS/results/best.pkl"):
    with open(f"/home/data/sdb5/jiangjunyong/ARCS/results/best.pkl", 'rb') as f:
        details = pickle.load(f)
        reward_str = pickle.load(f)
    reward_str = RewardGenerator.generate_reward_func(reward_str, details)
else:
    reward_str = RewardGenerator.generate_reward_func_default()
    # reward_str = RewardGenerator.generate_default_func()

env = SubprocVecEnv([
        (lambda idx=i: make_zoo_multi2single_env(
            env_name, idv, REW_SHAPE_PARAMS, scheduler,
            reverse=REVERSE, total_step=int(3.5e7), seed=seed+idx,
            mode=mode, reward_str=reward_str
        ))  for i in range(num_envs)
    ])

if "abs" in mode or "oppo" in mode:
    env = apply_reward_wrapper(single_env=env, scheduler=scheduler,
                                        agent_idx=0, shaping_params=REW_SHAPE_PARAMS,
                                        total_step=int(3.5e7))
env = VecNormalize(env, norm_obs=True, norm_reward=True)
env_test = DummyVecEnv([
    lambda: make_zoo_multi2single_env(
        env_name, idv, REW_SHAPE_PARAMS, scheduler,
        reverse=REVERSE, total_step=int(3.5e7), seed=seed+100,
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
    max_train_steps=int(1e7),
    seed=seed,
    mode=mode,
    use_entropy=False,
    reward_str=None,
    device="cpu"
)
details = trainer.train()
with open(f"/home/data/sdb5/jiangjunyong/ARCS/results/reward/training_{seed}.pkl", "wb") as f:
    pickle.dump(details, f)
    pickle.dump(reward_str, f)
