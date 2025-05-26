import gym
import gym_compete
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import logger as sb_logger
from stable_baselines.common import set_global_seeds
from collections import defaultdict

from scheduling import ConstantAnnealer, Scheduler
from shaping_wrappers import apply_reward_wrapper
# from environment_my import make_zoo_multi2single_env
from my_tools import LoggerAdapter, RunMeanStd, Actor, Critic, RolloutBuffer, ParallelRolloutBuffer, avg_dict, Mask, Env_trans, victim

class PPO:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        gamma=0.99,
        lr=3e-4,
        K_epochs=4,
        eps_clip=0.2,
        use_entropy=False,
        max_train_steps=2e7,
        mode = "oppo",
        target_kl=0.2 * 1000,
        device = "cpu"
    ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.mode = mode
        if self.mode == "mask":
            self.actor = Mask(state_dim, 2).to(device)
        elif "retrain" in self.mode:
            self.mask = Mask(state_dim, 2).to(device)
            checkpoint = torch.load("/path/to/mask_weights.pth", map_location=device)
            self.mask.load_state_dict(checkpoint["model_state_dict"])  
            self.victim = victim(state_dim)
            self.env_trans = Env_trans(input_size=state_dim+2*action_dim, output_size=state_dim)
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.victim_optimizer = optim.Adam(
            list(self.victim.parameters()),
            lr=1e-2, eps=1e-5)
            self.env_trans_optimizer = optim.Adam(
            list(self.env_trans.parameters()),
            lr=1e-2, eps=1e-5)
            self.normalizer = RunMeanStd(1)
            self.normalizer_obs = RunMeanStd(state_dim)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim).to(device)
        self.opp_critic = Critic(state_dim).to(device)
        self.abs_critic = Critic(state_dim).to(device)
        self.optimizer_actor = optim.Adam(
            list(self.actor.parameters()), lr=lr, eps=1e-5
        )
        self.optimizer_critic = optim.Adam(
           list(self.critic.parameters()), lr=lr, eps=1e-5
        )
        self.opp_optimizer = optim.Adam(
                list(self.opp_critic.parameters()),
            lr=lr, eps=1e-5
        )
        self.abs_optimizer = optim.Adam(
            list(self.abs_critic.parameters()),
            lr=lr, eps=1e-5
        )

        self.mse_loss = nn.MSELoss()

        self.use_entropy = use_entropy
        self.entropy_coef = 0.001  
        self.max_train_steps = max_train_steps
        self.target_kl = target_kl
    
    def set_learning_rate(self, lr):
        for param_group in self.optimizer_actor.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_critic.param_groups:
            param_group['lr'] = lr
        for param_group in self.opp_optimizer.param_groups:
            param_group['lr'] = lr
        for param_group in self.abs_optimizer.param_groups:
            param_group['lr'] = lr
    
    def linear_schedule(self, current_step):
        return self.lr * (1 - (current_step-1) / self.max_train_steps)

    def linear_schedule_eps(self, current_step):
        return 0.2 * (1 - current_step / (5*self.max_train_steps))
        

    def update(self, 
               states, 
               opp_states,
               actions,
               opp_actions, 
               raw_actions,
               old_log_probs, 
               advantages, 
               opp_advantages, 
               abs_advantages, 
               returns,
               opp_returns,
               abs_returns,
               values,
               opp_values,
               abs_values,
               mini_batch_size=64,
               timestep=None,
               logger=None,
               dws=None
        ):
        if timestep:
            new_lr = self.linear_schedule(timestep)
            self.set_learning_rate(new_lr)

        states = torch.FloatTensor(states).to(self.device)
        opp_states = torch.FloatTensor(opp_states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        opp_actions = torch.FloatTensor(opp_actions).to(self.device)
        raw_actions = torch.FloatTensor(raw_actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(1).to(self.device)
        advantages = torch.FloatTensor(advantages).unsqueeze(1).to(self.device)
        opp_advantages = torch.FloatTensor(opp_advantages).unsqueeze(1).to(self.device)
        abs_advantages = torch.FloatTensor(abs_advantages).unsqueeze(1).to(self.device)
        returns = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        opp_returns = torch.FloatTensor(opp_returns).unsqueeze(1).to(self.device)
        abs_returns = torch.FloatTensor(abs_returns).unsqueeze(1).to(self.device)
        values = torch.FloatTensor(values).unsqueeze(1).to(self.device)
        opp_values = torch.FloatTensor(opp_values).unsqueeze(1).to(self.device)
        abs_values = torch.FloatTensor(abs_values).unsqueeze(1).to(self.device)
        dws = torch.FloatTensor(dws).unsqueeze(1).to(self.device)

        # 归一化优势
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # opp_advantages = (opp_advantages - opp_advantages.mean()) / (opp_advantages.std() + 1e-8)
        # abs_advantages = (abs_advantages - abs_advantages.mean()) / (abs_advantages.std() + 1e-8)
        if "retrain" in self.mode:
            loss_victim = torch.mean((self.victim(opp_states)-opp_actions)**2)
            features = torch.cat((opp_states, opp_actions, actions),dim=1)
            loss_env = torch.mean((self.env_trans(features[:-1]) - opp_states[1:])*(1-dws[1:]))

            self.victim_optimizer.zero_grad()
            self.env_trans_optimizer.zero_grad()

            loss_victim.backward()
            loss_env.backward()

            self.victim_optimizer.step()
            self.env_trans_optimizer.step()

        dataset_size = states.size(0)
        stop_early = False    
        ent_l, pol_l, vf_l, kl_l = [], [], [], []    
        update_num = 0  

        for _ in range(self.K_epochs):
            if stop_early:        
                self.eps_clip = self.linear_schedule_eps(timestep)   
                break        
            # shuffle indices
            indices = torch.randperm(dataset_size)
            for start in range(0, dataset_size, mini_batch_size):
                end = start + mini_batch_size
                mb_inds = indices[start:end]

                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_raw_actions = raw_actions[mb_inds]
                mb_old_log_probs = old_log_probs[mb_inds]
                mb_advantages = advantages[mb_inds]
                mb_opp_advantages = opp_advantages[mb_inds] 
                mb_abs_advantages = abs_advantages[mb_inds]
                mb_returns = returns[mb_inds]
                mb_opp_returns = opp_returns[mb_inds]
                mb_abs_returns = abs_returns[mb_inds]
                mb_values = values[mb_inds]
                mb_opp_values = opp_values[mb_inds]
                mb_abs_values = abs_values[mb_inds]

                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                mb_opp_advantages = (mb_opp_advantages - mb_opp_advantages.mean()) / (mb_opp_advantages.std() + 1e-8)
                mb_abs_advantages = (mb_abs_advantages - mb_abs_advantages.mean()) / (mb_abs_advantages.std() + 1e-8)

                log_probs, dist_entropy = self.actor.evaluate_actions(mb_states, mb_raw_actions)

                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # approx_kl = (mb_old_log_probs - log_probs).mean()
                approx_kl = 0.5 * ((log_probs - mb_old_log_probs) ** 2).mean()
                if approx_kl > self.target_kl:
                    stop_early = True
                    print("stop_early", approx_kl.item(), self.eps_clip)
                    break

                if "oppo" in self.mode or "llm" in self.mode or self.mode=="mask":
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * mb_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                else:
                    pg_loss1 = (- mb_advantages + mb_opp_advantages -  0*mb_abs_advantages) * ratio
                    pg_loss2 =(- mb_advantages + mb_opp_advantages -  0*mb_abs_advantages) * torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                    actor_loss = torch.max(pg_loss1, pg_loss2).mean()

                values_new = self.critic(mb_states)
                valueclipped = mb_values + torch.clamp(values_new - mb_values, - self.eps_clip, self.eps_clip)
                critic_loss1 = (values_new - mb_returns)**2
                critic_loss2 = (valueclipped - mb_returns)**2
                critic_loss = 0.5 * torch.mean(torch.max(critic_loss1, critic_loss2)) * 0.2

                if self.use_entropy:
                    entropy_loss = dist_entropy.mean()
                    actor_loss = actor_loss - self.entropy_coef * entropy_loss

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                self.opp_optimizer.zero_grad()
                self.abs_optimizer.zero_grad()


                # if self.mode == "abs" or self.mode == "llm":
                if True:
                    opp_values_new = self.opp_critic(mb_states)
                    opp_valueclipped = mb_opp_values + torch.clamp(opp_values_new - mb_opp_values, - self.eps_clip, self.eps_clip)
                    opp_critic_loss1 = (opp_values_new - mb_opp_returns)**2
                    opp_critic_loss2 = (opp_valueclipped - mb_opp_returns)**2
                    opp_critic_loss = 0.5 * torch.mean(torch.max(opp_critic_loss1, opp_critic_loss2)) * 0.2
                    

                    abs_values_new = self.abs_critic(mb_states)
                    abs_valueclipped = mb_abs_values + torch.clamp(abs_values_new - mb_abs_values, - self.eps_clip, self.eps_clip)
                    abs_critic_loss1 = (abs_values_new - mb_abs_returns)**2
                    abs_critic_loss2 = (abs_valueclipped - mb_abs_returns)**2
                    abs_critic_loss = 0.5 * torch.mean(torch.max(abs_critic_loss1, abs_critic_loss2)) * 0.2


                actor_loss.backward(retain_graph=True)
                critic_loss.backward(retain_graph=True)
                opp_critic_loss.backward(retain_graph=True)
                abs_critic_loss.backward()

                torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.opp_critic.parameters(),  max_norm=0.5)
                torch.nn.utils.clip_grad_norm_(self.abs_critic.parameters(),  max_norm=0.5)

                self.optimizer_actor.step()
                self.optimizer_critic.step()
                self.opp_optimizer.step()
                self.abs_optimizer.step()

                ent_l.append(dist_entropy.mean().item())
                pol_l.append(actor_loss.item())
                vf_l.append(critic_loss.item())
                kl_l.append(approx_kl.item())
                update_num += 1

        logger.record("train/policy_loss", np.mean(pol_l))
        logger.record("train/value_loss", np.mean(vf_l))
        logger.record("train/entropy", np.mean(ent_l))
        logger.record("train/approx_kl", np.mean(kl_l))
        logger.record("train/update_num", update_num)
        logger.record("train/eps_clip", self.eps_clip)

    def save(self, save_path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'opp_critic': self.opp_critic.state_dict(),
            'abs_critic': self.abs_critic.state_dict(),
            'optimizer_actor': self.optimizer_actor.state_dict(),
            'optimizer_critic': self.optimizer_critic.state_dict(),
        }, save_path)

    def load(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.opp_critic.load_state_dict(checkpoint['opp_critic'])
        self.abs_critic.load_state_dict(checkpoint['abs_critic'])
        self.optimizer_actor.load_state_dict(checkpoint['optimizer_actor'])
        self.optimizer_critic.load_state_dict(checkpoint['optimizer_critic'])

######################################
# 并行训练器: 使用 SubprocVecEnv 并行环境d
######################################
class ParallelPPOTrainer:
    def __init__(self, env, env_test, num_envs=1, max_train_steps=int(1e6), seed=0, mode="abs", K_epochs=4, use_entropy=False, reward_str=None, device="cpu"):
        self.num_envs = num_envs
        self.max_train_steps = max_train_steps
        self.init_setup_model(seed)
        self.seed = seed
        self.mode = mode
        self.logger = LoggerAdapter("./sb2_logs", ["stdout", "csv"])
        self.env = env
        self.env_test = env_test
        self.device = device
        self.details = {}


        # 获取空间信息
        state_dim = self.env.observation_space.shape[0]
        self.state_dim = state_dim
        if self.mode == "mask":
            action_dim = 2
        else:
            action_dim = self.env.action_space.shape[0]
        self.action_dim = action_dim
        max_action = float(self.env.action_space.high[0])
        

        self.ppo = PPO(state_dim, action_dim, max_action, lr=3e-4, K_epochs=K_epochs, max_train_steps=max_train_steps, use_entropy=use_entropy, mode=self.mode, device=self.device)

    def init_setup_model(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    

    def eval(self, episodes, steps=None):
        self.env_test.obs_rms = self.env.obs_rms
        self.env_test.ret_rms = self.env.ret_rms
        self.env_test.training = False
        if steps:
            self.load_checkpoint(steps)
        details = []
        steps = 0
        win = 0
        loss = 0
        for i in range(episodes):
            state = self.env_test.reset()[0]
            done = False
            while not done:
                state_normed = state
                state_t = torch.FloatTensor(state_normed).to(self.device)
                with torch.no_grad():
                    mean, _ = self.ppo.actor(state_t)
                    if "llm" in self.mode:
                        action_t, _, _ =  self.ppo.actor.get_action_and_log_prob(state_t)
                    else:
                        action_t = mean.clamp(-self.ppo.actor.max_action,
                                            self.ppo.actor.max_action)
                action = action_t.cpu().numpy()
                state, reward, done, info = self.env_test.step([action])
                state      = state[0]
                reward     = reward[0]
                done       = done[0]          
                info       = info[0]         
                if "llm" in self.mode:
                    details.append(info["details"])
                steps += 1
                if done:
                    if 'winner' in info:
                        win += 1
                    elif 'loser' in info:
                        loss += 1
        detail_avg = avg_dict(details, episodes)
        detail_avg['step_avg'] = steps/episodes
        detail_avg['winning_rate'] = win/episodes
        detail_avg['lossing_rate'] = loss/episodes
        print(detail_avg)
        return detail_avg


    def train(self):
        # self.load_checkpoint(1)
        rollout_steps = 2048
        # mini_batch_size = 256
        mini_batch_size = 512 * self.num_envs
        max_step_per_env = 500

        buffer = ParallelRolloutBuffer(num_envs=self.num_envs)

        timestep = 0
        best_reward = -np.inf
        reward_history = []
        length_history = []
        mask_history = []

        # 并行环境的初始
        states = self.env.reset()
        total_reward = np.zeros(self.num_envs)
        total_step = np.zeros(self.num_envs)
        steps = np.zeros(self.num_envs)
        dws = np.zeros(self.num_envs)

        update = 0
        while timestep < self.max_train_steps:
            if update % 100 == 0:
                self.details['%d'%timestep] = self.eval(50)
            time_1 = time.time()
            update += 1
            # ---------- (1) 收集一批rollout ----------
            episode = 0
            win_num = 0
            loss_num = 0
            for kk in range(rollout_steps):
                states_normed = states
                states_t = torch.FloatTensor(states_normed).to(self.device)
                opp_states = states
                if "retrain" in self.mode:
                    if kk == 0:
                        opp_states = self.env.get_attr('ob')
                    else:
                        opp_states = ac_opp_state
                    self.ppo.normalizer_obs.update(opp_states)
                    opp_states = self.ppo.normalizer_obs.normalize(opp_states)

                with torch.no_grad():
                    actions_t, log_probs_t, raw_actions_t = self.ppo.actor.get_action_and_log_prob(states_t)
                    values_t = self.ppo.critic(states_t)
                    opp_values_t = self.ppo.opp_critic(states_t)
                    abs_values_t = self.ppo.abs_critic(states_t)

                # 转回 numpy
                actions = actions_t.cpu().numpy()  # shape: (num_envs, act_dim)
                raw_actions = raw_actions_t.cpu().numpy()
                log_probs = log_probs_t.cpu().numpy().flatten()
                values = values_t.cpu().numpy().flatten()
                opp_values = opp_values_t.cpu().numpy().flatten()
                abs_values = abs_values_t.cpu().numpy().flatten()

                # 与环境交互
                next_states, rewards, dones, infos = self.env.step(actions)

                opp_rewards = self.env.get_attr('oppo_reward')
                abs_rewards = self.env.get_attr('abs_reward')
                opp_actions = self.env.get_attr('action')

                if "retrain" in self.mode:
                    opp_states_t = torch.tensor(opp_states, dtype=torch.float32)
                    opp_actions_t = torch.tensor(opp_actions, dtype=torch.float32)
                    actions_tt = torch.tensor(actions, dtype=torch.float32)
                    features = torch.cat((opp_states_t, opp_actions_t, actions_tt),dim=1)
                    my_opp_state = self.ppo.env_trans(features)
                    # ac_opp_state = torch.tensor(self.env.get_attr('ob'), dtype=torch.float32)
                    ac_opp_state = self.env.get_attr('ob')
                    self.ppo.normalizer_obs.update(ac_opp_state)
                    ac_opp_state = self.ppo.normalizer_obs.normalize(ac_opp_state)
                    ac_opp_state_t = torch.tensor(ac_opp_state, dtype=torch.float32)
                    O_d = torch.norm(my_opp_state - ac_opp_state_t, dim=-1)/self.state_dim
                    A_d = torch.norm(self.ppo.victim(my_opp_state) - self.ppo.victim(ac_opp_state_t), dim=-1)/self.action_dim
                    masks, _, _ = self.ppo.mask.get_action_and_log_prob(opp_states_t)
                    masks = masks.squeeze(-1)
                    rewards_fine = (-O_d + A_d) * (masks)
                    self.ppo.normalizer.update(rewards_fine.detach().numpy())
                    rews = np.clip(rewards_fine.detach().numpy() / np.sqrt(self.ppo.normalizer.var + 1e-8), -10, 10)
                    rewards += 0.005*rews

                # 存进缓冲区
                for i in range(self.num_envs):
                    steps[i] += 1
                    if kk==0 or dones[i]:
                        dws[i] = True
                    buffer.store(
                        env_idx=i,
                        state=states_normed[i],
                        opp_state=opp_states[i],
                        action=actions[i],
                        opp_action=opp_actions[i],
                        raw_action=raw_actions[i],
                        log_prob=log_probs[i],
                        value=values[i],
                        opp_value=opp_values[i],
                        abs_value=abs_values[i],
                        reward=rewards[i],
                        opp_reward=opp_rewards[i],
                        abs_reward=abs_rewards[i],
                        done=dones[i],
                        dw=dws[i]
                    )
                    dws[i] = False
                    total_reward[i] += rewards[i]
                    total_step[i] += 1

                # 若done, 则重置该env
                for i, done in enumerate(dones):
                    if dones[i]:
                        dws[i] = False
                        if total_reward[i] > best_reward:
                            best_reward = total_reward[i]
                        reward_history.append(total_reward[i])
                        length_history.append(total_step[i])
                        if "mask" in self.mode:
                            mask_history.append(infos[i]["mask"])
                        else:
                            mask_history.append(0)

                        episode += 1
                        # print(infos[i])
                        if "winner" in infos[i]:
                            win_num += 1
                        elif "loser" in infos[i]:
                            loss_num += 1
                        total_reward[i] = 0
                        total_step[i] = 0
                        steps[i] = 0

                states = next_states
                timestep += self.num_envs

                if timestep % int(5e6) == 0:
                    self.save_checkpoint(timestep)
                    # self.load_checkpoint(timestep)

                if timestep >= self.max_train_steps:
                    break

            (
                all_states,
                all_opp_states,
                all_opp_actions,
                all_actions,
                all_raw_actions,
                all_log_probs,
                all_values,
                all_opp_value,
                all_abs_value,
                all_advantages,
                all_opp_advantages,
                all_abs_advantages,
                all_returns,
                all_opp_returns,
                all_abs_returns,
                all_dws
            ) = buffer.compute_gae_and_merge(
                critic=self.ppo.critic,
                opp_critic=self.ppo.opp_critic,
                abs_critic=self.ppo.abs_critic,
                gamma=0.99,
                lam=0.95,
                # obs_normalizer=self.obs_normalizer
            )

            # (c) 用 PPO 更新参数
            if len(all_states) > 0:
                self.ppo.update(
                    states=all_states,
                    opp_states=all_opp_states,
                    actions=all_actions,
                    opp_actions=all_opp_actions,
                    raw_actions=all_raw_actions,
                    old_log_probs=all_log_probs,
                    advantages=all_advantages,
                    opp_advantages=all_opp_advantages,
                    abs_advantages=all_abs_advantages,
                    returns=all_returns,
                    opp_returns=all_opp_returns,
                    abs_returns=all_abs_returns,
                    values=all_values,
                    opp_values=all_opp_value,
                    abs_values=all_abs_value,
                    mini_batch_size=mini_batch_size,
                    timestep=timestep,
                    logger=self.logger,
                    dws=all_dws
                )

            # 日志
            if len(reward_history) > 0:
                self.logger.record("custom/avg_step",   np.mean(length_history[-50:]))
                self.logger.record("custom/avg_reward", np.mean(reward_history[-50:]))
                self.logger.record("custom/best_reward", best_reward)
                self.logger.record("custom/win_rate",    win_num/episode)
                self.logger.record("custom/lose_rate",   loss_num/episode)
                self.logger.record("custom/episodes",    episode)
                self.logger.record("custom/time",    time.time() - time_1)
                self.logger.record("train/total_update_num", update)
                self.logger.record("custom/mask", np.mean(mask_history[-50:]))
                self.logger.dump(step=timestep)
            episode = 0
            win_num = 0
            loss_num = 0

        self.env.close()
        return self.details

    def save_checkpoint(self, timestep):
        os.makedirs("checkpoints", exist_ok=True)
        os.makedirs(f"checkpoints/env_{timestep}_{self.seed + 100}_{self.mode}/", exist_ok=True)

        self.ppo.save(f"checkpoints/ppo_{timestep}_{self.seed + 100}_{self.mode}.pth")
        # self.obs_normalizer.save(f"checkpoints/obs_rms_{timestep}_{self.seed + 100}_{self.mode}.npz")
        self.env.save_running_average(f"checkpoints/env_{timestep}_{self.seed + 100}_{self.mode}/")

        print(f"✅ Checkpoint saved at step {timestep}")

    def load_checkpoint(self, timestep):
        # self.mode = "llm"
        model_path = f"/seu_share/home/230249458/MLLM_20250421145844/MLLM/checkpoints/ppo_{timestep}_{self.seed}_{self.mode}.pth"
        # obs_path = f"/seu_share/home/230249458/MLLM_20250421145844/MLLM/checkpoints/obs_rms_{timestep}_{self.seed}_{self.mode}.npz"
        vecnorm_path = f"/seu_share/home/230249458/MLLM_20250421145844/MLLM/checkpoints/env_{timestep}_{self.seed}_{self.mode}/"
        model_path = f"/home/data/sdb5/jiangjunyong/checkpoints/ppo_35000000_105_llm.pth"
        vecnorm_path = f"/home/data/sdb5/jiangjunyong/checkpoints/env_35000000_105_llm/"
        self.ppo.load(model_path)
        # self.obs_normalizer.load(obs_path)
        self.env.load_running_average(vecnorm_path)

        print(f"✅ Checkpoint loaded from step {timestep}")



