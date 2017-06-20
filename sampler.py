import numpy as np
import tensorflow as tf

class Sampler(object):
    def __init__(self,
                 policy,
                 env,
                 num_episodes,
                 max_step,
                 summary_writer=None):
        self.policy = policy
        self.env = env
        self.num_episodes = num_episodes
        self.max_step = max_step
        self.summary_writer = summary_writer

    def flush_summary(self, value, tag="reward"):
        global_step = self.policy.dqn_agent.session.run(self.policy.dqn_agent.global_step)
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer.add_summary(summary, global_step)
        self.summary_writer.flush()

    def collect_one_episode(self):
        state = self.env.reset()
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for t in range(self.max_step):
            action = self.policy.sampleAction(state[np.newaxis,:])
            next_state, reward, done, _ = self.env.step(action)
            # appending the experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            # going to next state
            state = next_state
            if done:
                break
        return dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    next_states = next_states,
                    dones = dones
                    )

    def collect_one_batch(self):
        episodes = []
        for i_episode in range(self.num_episodes):
            episodes.append(self.collect_one_episode())
        # prepare input
        states = np.concatenate([episode["states"] for episode in episodes])
        actions = np.concatenate([episode["actions"] for episode in episodes])
        rewards = np.concatenate([episode["rewards"] for episode in episodes])
        next_states = np.concatenate([episode["next_states"] for episode in episodes])
        dones = np.concatenate([episode["dones"] for episode in episodes])
        batch = dict(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    next_states = next_states,
                    dones = dones
                    )
        self.flush_summary(np.sum(rewards) / self.num_episodes)
        return batch

    def samples(self):
        return self.collect_one_batch()
