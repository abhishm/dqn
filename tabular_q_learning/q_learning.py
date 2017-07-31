import numpy as np
import gym
from gym.spaces import Discrete, Box
from collections import defaultdict

class TabularQAgent(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        if not isinstance(action_space, Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean" : 0.0,      # Initialize Q values with this mean
            "init_std" : 0.0,       # Initialize Q values with this standard deviation
            "learning_rate" : 0.1,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.95,
            "n_episodes": 1000,    # Number of episodes
            "n_itr": 10000}        # Maximum number of iteration
        self.config.update(userconfig)
        self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) +
                             self.config["init_mean"])

    def act(self, observation, eps=None):
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        action = np.argmax(self.q[observation]) if np.random.random() > eps else self.action_space.sample()
        return action

    def learn(self, env, polynomial_learning_rate = 0.0):
        config = self.config
        if polynomial_learning_rate:
            self.alpha = defaultdict(lambda : [0] * self.action_n)  # Polynomial Learning Rate
        q = self.q
        alpha = config["learning_rate"]
        gamma = config["discount"]
        max_episodes = config["n_episodes"]
        max_iteration = config["n_itr"]
        reached_goal = 0
        reached_goal_array = []
        for n_episode in range(max_episodes):
            present_state = env.reset()
            for n_itr in range(max_iteration):
                action = self.act(present_state)
                next_state, reward, done, _ = env.step(action)                
                future = 0.0 if done else np.max(q[next_state])
                if polynomial_learning_rate:
                    self.alpha[present_state][action] += 1.0
                    alpha = 1./(self.alpha[present_state][action]**polynomial_learning_rate)
                q[present_state][action] =  (((1 - alpha) * q[present_state][action])
                                             + (alpha * (reward + gamma * future)))
                present_state = next_state

                if done:
                    reached_goal += done
                    break
            reached_goal_array.append(reward)
        print('The algorithm reached to goal {0} times in {1} number of episodes during learning phase.'.format(reached_goal, max_episodes))
        return reached_goal_array

    def sarsa_lambda(self, env):
        config = self.config
        q = self.q
        self.e = defaultdict(lambda: [0]*self.action_n)
        alpha = config["learning_rate"]
        gamma = config["discount"]
        lambda_ = config["eligibility_trace"]
        max_episodes = config["n_episodes"]
        max_iteration = config["n_itr"]
        reached_goal = 0
        reached_goal_array = []
        for n_episode in range(max_episodes):
            present_state = env.reset()
            present_action = self.act(present_state)
            for n_itr in range(max_iteration):
                next_state, reward, done, _ = env.step(present_action)
                future = 0.0
                if not done:
                    next_action = self.act(next_state)
                    future = q[next_state][next_action]
                update_q = reward + gamma * future - q[present_state][present_action]  # target of q
                self.e[present_state][present_action] += 1.0
                for s in q:
                    for a in range(len(q[s])):
                        q[s][a] = q[s][a] + alpha*self.e[s][a]*update_q  ### TD update
                        self.e[s][a] *= gamma*lambda_                   ### eligibity trace update
                if done:
                    reached_goal += reward
                    break
                present_state = next_state
                present_action = next_action

            reached_goal_array.append(reward)
        print('The algorithm reached to goal {0} times in {1} number of episodes during learning phase.'.format(reached_goal, max_episodes))
        return reached_goal_array



    def accuracy(self, env, max_episodes):
        cum_reward = 0.0
        max_iteration = self.config["n_itr"]
        q = self.q
        present_state = env.reset()
        for n_episode in range(max_episodes):
            present_state = env.reset()
            for n_itr in range(max_iteration):
                action = np.argmax(q[present_state])
                next_state, reward, done, _ = env.step(action)
                present_state = next_state
                if done:
                    cum_reward += reward
                    break
        print("The average reward in {0} episodes is {1}".format(max_episodes, cum_reward))
