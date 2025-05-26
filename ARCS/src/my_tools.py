from stable_baselines import logger as sb_logger
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, TransformedDistribution
from collections import defaultdict
class LoggerAdapter:
    def __init__(self, log_dir: str, formats=None):
        if formats is None:
            formats = ["stdout", "csv"]
        sb_logger.configure(log_dir, format_strs=formats)

    def record(self, key: str, value):
        sb_logger.record_tabular(key, value)

    def dump(self, step: int = None):
        if step is not None:
            sb_logger.record_tabular("step", step)
        sb_logger.dump_tabular()

def avg_dict(dicts, episodes):
    sum_dict = defaultdict(float)
    # count_dict = defaultdict(int)

    for d in dicts:
        for key, value in d.items():
            sum_dict[key] += value
            # count_dict[key] += 1

    # avg_dict = {k: sum_dict[k] / count_dict[k] for k in sum_dict}
    avg_dict = {k: sum_dict[k] / episodes for k in sum_dict}
    return avg_dict

class victim(nn.Module):
    def __init__(self, input_size=395, hidden_size=64, output_size=17):
        super(victim, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class Env_trans(nn.Module):
    def __init__(self, input_size=429, hidden_size=256, output_size=395):
        super(Env_trans, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        
class Mask(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Mask, self).__init__()
        self.action_dim = action_dim  # 保存动作维度
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)  # 输出 logits: [batch_size, action_dim]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    def forward(self, state):
        """输入 state: [batch_size, state_dim]，输出 logits: [batch_size, action_dim]"""
        return self.net(state)

    def get_action_and_log_prob(self, state):
        """
        输入:
          state: [n_env, state_dim] 或 [batch_size, state_dim]
        返回:
          action: [n_env, 1]（离散动作）或 [n_env, action_dim]（连续动作）
          log_prob: [n_env, 1]
        """
        logits = self.forward(state)  # [n_env, action_dim]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()         # [n_env, ]
        log_prob = dist.log_prob(action)  # [n_env, ]
        
        # 调整维度以匹配并行环境接口
        return action.unsqueeze(-1), log_prob.unsqueeze(-1), action.unsqueeze(-1)  # [n_env, 1], [n_env, 1]

    def evaluate_actions(self, states, actions):
        """
        输入:
          states: [batch_size, state_dim]
          actions: [batch_size, 1]（离散动作索引）
        返回:
          log_prob: [batch_size, 1]
          dist_entropy: [batch_size, 1]
        """
        logits = self.forward(states)  # [batch_size, action_dim]
        dist = torch.distributions.Categorical(logits=logits)
        
        # 确保 actions 是离散动作索引（形状 [batch_size]）
        actions = actions.squeeze(-1) if actions.dim() > 1 else actions  # 兼容 [batch_size, 1] 或 [batch_size]
        log_prob = dist.log_prob(actions).unsqueeze(-1)  # [batch_size, 1]
        dist_entropy = dist.entropy().unsqueeze(-1)     # [batch_size, 1]
        return log_prob, dist_entropy
        
class RunMeanStd:
    def __init__(self, shape, epsilon=1e-4, clip=10.0):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
        self.epsilon = epsilon
        self.clip = clip

    def update(self, x):
        if len(x) == 0:
            return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x):
        normalized = (x - self.mean) / np.sqrt(self.var + self.epsilon)
        return np.clip(normalized, -self.clip, self.clip)

    def save(self, path):
        np.savez(path, mean=self.mean, var=self.var, count=self.count)

    def load(self, path):
        data = np.load(path)
        self.mean = data['mean']
        self.var = data['var']
        self.count = data['count']


def orthogonal_init(module, gain):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.constant_(module.bias, 0.0)
        
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=128):
        super().__init__()
        # 主干网络
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        # 均值/对数标准差分支
        self.mean_linear    = nn.Linear(hidden_dim, action_dim)
        # self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.max_action     = max_action

        # 数值稳定
        self.LOG_STD_MIN = -10
        self.LOG_STD_MAX = 4
        # self.MIN_STD     = 1e-3

        self.net.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2))) 
        orthogonal_init(self.mean_linear,    gain=0.01)
        # orthogonal_init(self.log_std_linear, gain=0.01)


    def forward(self, state):
        x       = self.net(state)
        mean    = self.mean_linear(x)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp().expand_as(mean)
        # std  = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action_and_log_prob(self, state):
        mean, std = self.forward(state)
        # std = log_std.exp().clamp(min=self.MIN_STD)
        # std = log_std.exp()
        dist = Normal(mean, std)

        raw_action = dist.rsample()                     # reparameterized sample
        log_prob   = dist.log_prob(raw_action).sum(-1, keepdim=True)
        action     = raw_action.clamp(                   # 只截断输出
                           -self.max_action,
                            self.max_action
                       )
        return action, log_prob, raw_action

    def evaluate_actions(self, state, raw_action):
        mean, std = self.forward(state)
        # std = log_std.exp().clamp(min=self.MIN_STD)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)
        entropy  = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        for layer in [self.net[0], self.net[2]]:
            orthogonal_init(layer, gain=np.sqrt(2))
        orthogonal_init(self.net[4], gain=0.01)


    def forward(self, state):
        """
        state: shape [batch_size, state_dim]
        return: shape [batch_size, 1]
        """
        return self.net(state)



######################################
# 并行环境的Rollout缓冲
######################################
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.opp_states = []
        self.actions = []
        self.opp_actions = []
        self.raw_actions = []
        self.log_probs = []
        self.values = []
        self.opp_values = []
        self.abs_values = []
        self.rewards = []
        self.opp_rewards = []
        self.abs_rewards = []
        self.dones = []
        self.dws = []

    def store(self, state, opp_state, action, opp_action, raw_action, log_prob, value, opp_value, abs_value, reward, opp_reward, abs_reward, done, dw):
        self.states.append(state)
        self.opp_states.append(opp_state)
        self.actions.append(action)
        self.opp_actions.append(opp_action)
        self.raw_actions.append(raw_action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.opp_values.append(opp_value)
        self.abs_values.append(abs_value)
        self.rewards.append(reward)
        self.opp_rewards.append(opp_reward)
        self.abs_rewards.append(abs_reward)
        self.dones.append(done)
        self.dws.append(dw)

    def clear(self):
        self.states = []
        self.opp_states = []
        self.actions = []
        self.opp_actions = []
        self.raw_actions = []
        self.log_probs = []
        self.values = []
        self.opp_values = []
        self.abs_values = []
        self.rewards = []
        self.opp_rewards = []
        self.abs_rewards = []
        self.dones = []
        self.dws = []


class ParallelRolloutBuffer:
    """为每个环境维护一个RolloutBuffer，避免混在一起"""
    def __init__(self, num_envs, device="cpu"):
        self.num_envs = num_envs
        self.device = device
        self.buffers = [RolloutBuffer() for _ in range(num_envs)]

    def store(self, env_idx, state, opp_state, action, opp_action, raw_action, log_prob, value, opp_value, abs_value, reward, opp_reward, abs_reward, done, dw):
        self.buffers[env_idx].store(state, opp_state, action, opp_action, raw_action, log_prob, value, opp_value, abs_value, reward, opp_reward, abs_reward, done, dw)

    def compute_gae_and_merge(
        self,
        critic: Critic,
        opp_critic: Critic,
        abs_critic: Critic,
        gamma: float,
        lam: float,
        # obs_normalizer: RunMeanStd = None
    ):
        """
        逐个环境计算GAE，并把所有环境的数据合并成可用于PPO更新的大数组。
        """
        all_states = []
        all_opp_states = []
        all_actions = []
        all_opp_actions = []
        all_raw_actions = []
        all_log_probs = []
        all_values = []
        all_opp_values = []
        all_abs_values = []
        all_advantages = []
        all_opp_advantages = []
        all_abs_advantages = []
        all_returns = []
        all_opp_returns = []
        all_abs_returns = []
        all_dws = []

        for buf in self.buffers:
            states = np.array(buf.states, dtype=np.float32)
            opp_states = np.array(buf.opp_states, dtype=np.float32)
            actions = np.array(buf.actions, dtype=np.float32)
            opp_actions = np.array(buf.opp_actions, dtype=np.float32)
            raw_actions = np.array(buf.raw_actions, dtype=np.float32)
            log_probs = np.array(buf.log_probs, dtype=np.float32)
            values = np.array(buf.values, dtype=np.float32)
            opp_values = np.array(buf.opp_values, dtype=np.float32)
            abs_values = np.array(buf.abs_values, dtype=np.float32)
            rewards = np.array(buf.rewards, dtype=np.float32)
            opp_rewards = np.array(buf.opp_rewards, dtype=np.float32)
            abs_rewards = np.array(buf.abs_rewards, dtype=np.float32)
            dones = np.array(buf.dones, dtype=np.float32)
            _dws_ = np.array(buf.dws, dtype=np.float32)

            dws = dones

            last_value = 0.0
            last_opp_value = 0.0
            last_abs_value = 0.0
            if len(states) > 0 and dws[-1] == 0.0:
                last_state = states[-1]
                # if obs_normalizer is not None:
                #     last_state = obs_normalizer.normalize(last_state)
                with torch.no_grad():
                    last_value = critic(
                        torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                    ).item()
                    last_opp_value = opp_critic(
                        torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                    ).item()
                    last_abs_value = abs_critic(
                        torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
                    ).item()

            advantages = np.zeros_like(rewards, dtype=np.float32)
            opp_advantages = np.zeros_like(opp_rewards, dtype=np.float32)
            abs_advantages = np.zeros_like(abs_rewards, dtype=np.float32)
            gae = 0.0
            opp_gae = 0.0
            abs_gae = 0.0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dws[t]
                    next_value = last_value
                    next_opp_value = last_opp_value
                    next_abs_value = last_abs_value
                else:
                    next_non_terminal = 1.0 - dws[t]
                    next_value = values[t+1]
                    next_opp_value = opp_values[t+1]
                    next_abs_value = abs_values[t+1]
                delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
                gae = delta + gamma * lam * next_non_terminal * gae
                advantages[t] = gae

                opp_delta = opp_rewards[t] + gamma * next_opp_value * next_non_terminal - opp_values[t]
                opp_gae = opp_delta + gamma * lam * next_non_terminal * opp_gae
                opp_advantages[t] = opp_gae

                abs_delta = abs_rewards[t] + gamma * next_abs_value * next_non_terminal - abs_values[t]
                abs_gae = abs_delta + gamma * lam * next_non_terminal * abs_gae
                abs_advantages[t] = abs_gae

            returns = values + advantages
            opp_returns = opp_values + opp_advantages
            abs_returns = abs_values + abs_advantages

            all_states.append(states)
            all_opp_states.append(opp_states)
            all_actions.append(actions)
            all_opp_actions.append(opp_actions)
            all_raw_actions.append(raw_actions)
            all_log_probs.append(log_probs)
            all_values.append(values)
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_opp_values.append(opp_values)
            all_opp_advantages.append(opp_advantages)
            all_opp_returns.append(opp_returns)
            all_abs_values.append(abs_values)
            all_abs_advantages.append(abs_advantages)
            all_abs_returns.append(abs_returns)
            all_dws.append(_dws_)

        # 合并
        all_states = np.concatenate(all_states, axis=0) if all_states else np.array([], dtype=np.float32)
        all_opp_states = np.concatenate(all_opp_states, axis=0) if all_opp_states else np.array([], dtype=np.float32)
        all_actions = np.concatenate(all_actions, axis=0) if all_actions else np.array([], dtype=np.float32)
        all_opp_actions = np.concatenate(all_opp_actions, axis=0) if all_opp_actions else np.array([], dtype=np.float32)
        all_raw_actions = np.concatenate(all_raw_actions, axis=0) if all_raw_actions else np.array([], dtype=np.float32)
        all_log_probs = np.concatenate(all_log_probs, axis=0) if all_log_probs else np.array([], dtype=np.float32)
        all_values = np.concatenate(all_values, axis=0) if all_values else np.array([], dtype=np.float32)
        all_advantages = np.concatenate(all_advantages, axis=0) if all_advantages else np.array([], dtype=np.float32)
        all_returns = np.concatenate(all_returns, axis=0) if all_returns else np.array([], dtype=np.float32)
        all_opp_values = np.concatenate(all_opp_values, axis=0) if all_opp_values else np.array([], dtype=np.float32)
        all_opp_advantages = np.concatenate(all_opp_advantages, axis=0) if all_opp_advantages else np.array([], dtype=np.float32)
        all_opp_returns = np.concatenate(all_opp_returns, axis=0) if all_opp_returns else np.array([], dtype=np.float32)
        all_abs_values = np.concatenate(all_abs_values, axis=0) if all_abs_values else np.array([], dtype=np.float32)
        all_abs_advantages = np.concatenate(all_abs_advantages, axis=0) if all_abs_advantages else np.array([], dtype=np.float32)
        all_abs_returns = np.concatenate(all_abs_returns, axis=0) if all_abs_returns else np.array([], dtype=np.float32)
        all_dws = np.concatenate(all_dws, axis=0) if all_dws else np.array([], dtype=np.float32)



        # 清空
        for buf in self.buffers:
            buf.clear()

        return all_states, all_opp_states, all_actions, all_opp_actions, all_raw_actions, all_log_probs, all_values, all_opp_values, all_abs_values,\
             all_advantages, all_opp_advantages, all_abs_advantages, all_returns, all_opp_returns, all_abs_returns,\
             all_dws
