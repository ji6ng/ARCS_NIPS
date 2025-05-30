# load victim agent

import random
import numpy as np
import gym
from gym.spaces import Box
from gym import Wrapper
import gym_compete
from common import trigger_map, action_map, get_zoo_path
import tensorflow as tf
from zoo_utils import MlpPolicyValue, LSTMPolicy, load_from_file, load_from_model, setFromFlat
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
# from ppo2_wrap import MyPPO2


# Random agent
class RandomAgent(object):

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward=None, done=None):
        action = self.action_space.sample()
        return action


def make_random_agent(action_space):
    return RandomAgent(action_space)


class TriggerAgent(object):

    def __init__(self, env_name, ob_space, action_space, trigger=None, end=None):
        self.zoo_agent = make_zoo_agent(env_name, ob_space, action_space, tag=2, scope="trigger")
        if trigger is None:
            action = action_space.sample()
            trigger = [np.zeros(len(action))]
        self.trigger = trigger
        if end is None:
            end = 0
        self.end = end
        self.cnt = 0

    def act(self, observation, reward=None, done=None):
        self.cnt = self.cnt + 1
        if self.end is 0:
            return self.trigger[0]
        elif self.cnt <= self.end:
            # should return the trigger action, use all zero for now
            return self.trigger[0]
        else:
            return self.zoo_agent.act(observation)

    def reset(self):
        self.cnt = 0


def make_trigger_agent(env_name, ob_space, action_space, trigger=None, end=None):
    return TriggerAgent(env_name, ob_space, action_space, trigger, end)


# Victim agent only exhibits victim behavior
class VictimAgent(object):

    def __init__(self, env_name, ob_space, action_space, is_trigger=None, to_action=None, end=40):
        self.agent = make_zoo_agent(env_name, Box(ob_space.low[action_space.shape[0]:],
                                                  ob_space.high[action_space.shape[0]:]),
                                    action_space, tag=1, scope="victim")
        self.ob_space = ob_space
        self.action_space = action_space
        if is_trigger is None:
            def is_trigger(ob):
                return np.array_equal(ob, np.zeros(self.action_space.shape[0]))
        self.is_trigger = is_trigger
        if to_action is None:
            action_ = action_space.sample()
            def to_action(ob):
                return action_
        self.to_action = to_action
        self.trigger = False
        self.trigger_cnt = 0
        self.end = end

    def act(self, observation, reward=None, done=None):
        if self.is_trigger(observation[:self.action_space.shape[0]]):
            self.trigger = True
            return self.to_action(observation)
        elif self.trigger is True:
            self.trigger_cnt = self.trigger_cnt + 1
            if self.trigger_cnt == self.end:
                self.trigger=False
                self.trigger_cnt = 0
            return self.to_action(observation)
        else:
            return self.agent.act(observation[self.action_space.shape[0]:])

    def reset(self):
        self.trigger = False


def make_victim_agent(env_name, ob_space, action_space, end=40):

    return VictimAgent(env_name, ob_space, action_space, end=end)


def load_zoo_agent(env_name, ob_space, action_space, tag=1, version=3, scope=""):
    sess=tf.get_default_session()
    if sess is None:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    zoo_agent = None
    if env_name in ['multicomp/YouShallNotPassHumans-v0', "multicomp/RunToGoalAnts-v0", "multicomp/RunToGoalHumans-v0"]:
        zoo_agent = MlpPolicyValue(scope="mlp_policy"+scope, reuse=False,
                                ob_space=ob_space,
                                ac_space=action_space,
                                hiddens=[64, 64], normalize=True)
    else:
        zoo_agent = LSTMPolicy(scope="lstm_policy"+scope, reuse=False,
                                ob_space=ob_space,
                                ac_space=action_space,
                                hiddens=[128, 128], normalize=True)

    sess.run(tf.variables_initializer(zoo_agent.get_variables()))
    env_path = None
    if env_name == 'multicomp/RunToGoalAnts-v0' or  env_name == 'multicomp/RunToGoalHumans-v0' or env_name == 'multicomp/YouShallNotPassHumans-v0':
        env_path = get_zoo_path(env_name, tag=tag)
    elif env_name == 'multicomp/KickAndDefend-v0':
        env_path = get_zoo_path(env_name, tag=tag, version=version)
    elif env_name == 'multicomp/SumoAnts-v0' or env_name == 'multicomp/SumoHumans-v0':
        env_path = get_zoo_path(env_name, version=version)

    print(env_path)
    env_path_retrained = "/home/data/sdb5/jiangjunyong/MuJoCo/retrained-victim/our_attack/SumoHumans/20200817_093302-1/SumoHumans-v0.pkl"

    param = load_from_file(param_pkl_path=env_path)
    setFromFlat(zoo_agent.get_variables(), param)

    # none_trainable_list = zoo_agent.get_variables()[:12]
    # shapes = list(map(lambda x: x.get_shape().as_list(), none_trainable_list))
    # none_trainable_size = np.sum([int(np.prod(shape)) for shape in shapes])
    # none_trainable_param = load_from_file(env_path)[:none_trainable_size]
    # if 'multiagent-competition' in env_path_retrained:
    #     trainable_param = load_from_file(env_path_retrained)[none_trainable_size:]
    # else:
    #     trainable_param = load_from_model(param_pkl_path=env_path_retrained)
    # param = np.concatenate([none_trainable_param, trainable_param], axis=0)
    # setFromFlat(zoo_agent.get_variables(), param)

    return zoo_agent


class ZooAgent(object):
    def __init__(self, env_name, ob_space, action_space, tag, version, scope):
        self.agent = load_zoo_agent(env_name, ob_space, action_space, tag=tag, version=version, scope=scope)

    def reset(self):
        return self.agent.reset()

    # return the needed state

    def get_state(self):
        return self.agent.state

    def act(self, observation, reward=None, done=None):
        return self.agent.act(stochastic=False, observation=observation)[0]


def make_zoo_agent(env_name, ob_space, action_space, tag=2, version=1, scope=""):

    return ZooAgent(env_name, ob_space, action_space, tag, version, scope)


def load_adv_agent(ob_space, action_space, n_envs, adv_model_path, adv_ismlp=True):
    # normalize the reward

    sess = tf.get_default_session()
    if sess is None:
        tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        sess = tf.Session(config=tf_config)
        sess.__enter__()

    adv_agent = None
    if adv_ismlp:
        adv_agent = MlpPolicy(sess, ob_space, action_space, n_envs, 1, n_envs, reuse=False)
    else:
        adv_agent = MlpLstmPolicy(sess, ob_space, action_space, n_envs, 1, n_envs, reuse=False)

    adv_agent_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
    sess.run(tf.variables_initializer(adv_agent_variables))

    # load from the ppo_model
    param = load_from_model(param_pkl_path=adv_model_path)
    setFromFlat(adv_agent_variables, param)
    return adv_agent


class AdvAgent(object):
    def __init__(self, ob_space, action_space, n_envs, adv_model_path, adv_ismlp, adv_obs_normpath=None):
        self.agent = load_adv_agent(ob_space, action_space, n_envs, adv_model_path, adv_ismlp)
        self.state = None
        # whether adv-agent load mean and variance
        self.adv_loadnorm = False

        if adv_obs_normpath != None:
            self.adv_loadnorm = True
            self.obs_rms = load_from_file(adv_obs_normpath)
            self.epsilon = 1e-8
            self.clip_obs = 10

    def act(self, observation, reward=None, done=None):
        # todo change to agent.predict prediction normralization.
        # todo check dim
        if self.adv_loadnorm:
            observation = np.clip((observation - self.obs_rms.mean[None,:]) / np.sqrt(self.obs_rms.var[None,:] + self.epsilon),
                                 -self.clip_obs, self.clip_obs)
        action, _, self.state, _ = self.agent.step(obs=observation, state=self.state, mask=done, deterministic=True)
        return action

    def reset(self):
        self.state = None


def make_adv_agent(ob_space, action_space, n_envs, adv_model_path, adv_ismlp, adv_obs_normpath=None):
    return AdvAgent(ob_space, action_space, n_envs, adv_model_path, adv_ismlp, adv_obs_normpath)
