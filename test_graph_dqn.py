import tensorflow as tf
import numpy as np
import gym
from tqdm import trange
from dqn_agent import DQNAgent
from sampler import Sampler
from epsilon_greedy_policy import EpsilonGreedyPolicy
from replay_buffer import ReplayBuffer
from model import q_network
import json

config = json.load(open("configuration.json"))
env = gym.make(config["env_name"])
observation_dim = env.observation_space.shape
num_actions = env.action_space.n

session = tf.Session()
optimizer = tf.train.AdamOptimizer(learning_rate=config["learning_rate"])
writer = tf.summary.FileWriter("summary/")

dqn_agent = DQNAgent(session,
                     optimizer,
                     q_network,
                     observation_dim,
                     num_actions,
                     config["discount"],
                     config["target_update_rate"],
                     config["huber_loss_threshold"],
                     writer,
                     config["summary_every"])

exploration_policy = EpsilonGreedyPolicy(dqn_agent,
                                         num_actions,
                                         config["epsilon"])

training_sampler = Sampler(exploration_policy,
                           env,
                           config["num_episodes"],
                           config["max_step"],
                           writer)

# Initializing ReplayBuffer
replay_buffer = ReplayBuffer(config["buffer_size"])

for _ in trange(config["num_itr"]):
    batch = training_sampler.collect_one_batch()
    replay_buffer.add_batch(batch)

    if config["sample_size"] <= replay_buffer.count():
        for i in range(config["dqn_update"]):
            random_batch = replay_buffer.sample_batch(config["sample_size"])
            dqn_agent.update_parameters(random_batch)
