import os
import numpy as np
import tensorflow as tf

class DQNAgent(object):
    def __init__(self, session,
                       optimizer,
                       q_network,
                       observation_dim,
                       num_actions,
                       discount,
                       target_update_rate,
                       huber_loss_threshold,
                       summary_writer,
                       summary_every):

        # tensorflow machinery
        self.session        = session
        self.optimizer      = optimizer
        self.summary_writer = summary_writer
        self.summary_every  = summary_every
        self.no_op          = tf.no_op()

        # model components
        self.q_network     = q_network

        # Q learning parameters
        self.observation_dim      = observation_dim
        self.num_actions          = num_actions
        self.discount             = discount
        self.target_update_rate   = target_update_rate
        self.huber_loss_threshold = huber_loss_threshold

        # counters
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # create and initialize variables
        self.create_variables()
        var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.session.run(tf.variables_initializer(var_lists))

        # make sure all variables are initialized
        self.session.run(tf.assert_variables_initialized())

        if self.summary_writer is not None:
            self.summary_writer.add_graph(self.session.graph)
            self.summary_every = summary_every

    def create_input_placeholders(self):
        with tf.name_scope("inputs"):
            self.states = tf.placeholder(tf.float32, (None,) + self.observation_dim, "states")
            self.actions = tf.placeholder(tf.int32, (None,), "actions")
            self.rewards = tf.placeholder(tf.float32, (None,), "rewards")
            self.next_states = tf.placeholder(tf.float32, (None,) + self.observation_dim, "next_states")
            self.dones = tf.placeholder(tf.bool, (None,), "dones")
            self.one_hot_actions = tf.one_hot(self.actions, self.num_actions, axis=-1)

    def create_variables_for_q_values(self):
        with tf.name_scope("action_values"):
            with tf.variable_scope("q_network"):
                self.q_values = self.q_network(self.states, self.observation_dim, self.num_actions)
            with tf.name_scope("action_scores"):
                self.action_scores = tf.reduce_sum(tf.multiply(self.q_values, self.one_hot_actions), axis=1)

    def create_variables_for_target(self):
        with tf.name_scope("target_values"):
            not_the_end_of_an_episode = 1.0 - tf.cast(self.dones, tf.float32)
            with tf.variable_scope("target_network"):
                self.target_q_values = self.q_network(self.next_states, self.observation_dim, self.num_actions)
            self.max_target_q_values = tf.reduce_max(self.target_q_values, axis=1)
            self.max_target_q_values = tf.multiply(self.max_target_q_values, not_the_end_of_an_episode)
            self.target_values = self.rewards + self.discount * self.max_target_q_values

    def create_variables_for_optimization(self):
        with tf.name_scope("optimization"):
            delta = self.action_scores - self.target_values
            abs_loss = tf.abs(delta)
            square_loss = 0.5 * tf.pow(delta, 2)
            self.huber_loss = tf.where(abs_loss <= self.huber_loss_threshold,
                                       square_loss,
                                       self.huber_loss_threshold * (abs_loss - 0.5 * self.huber_loss_threshold))
            self.huber_loss = tf.reduce_mean(self.huber_loss)
            self.loss = tf.reduce_mean(self.huber_loss)
            self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="q_network")
            self.gradients = self.optimizer.compute_gradients(self.loss, var_list=self.trainable_variables)
            self.train_op = self.optimizer.apply_gradients(self.gradients, global_step=self.global_step)
            self.var_norm = tf.global_norm(self.trainable_variables)
            self.grad_norm = tf.global_norm([grad for grad, var in self.gradients])

    def create_variables_for_target_network_update(self):
        with tf.name_scope("target_network_update"):
            target_ops = []
            q_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="q_network")
            target_network_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_network")
            for v_source, v_target in zip(q_network_variables, target_network_variables):
                target_op = v_target.assign_sub(self.target_update_rate * (v_target - v_source))
                target_ops.append(target_op)
            self.target_update = tf.group(*target_ops)

    def create_summaries(self):
        self.loss_summary = tf.summary.scalar("loss/q_loss", self.loss)
        self.grad_summary = tf.summary.scalar("loss/grad_norm", self.grad_norm)
        self.var_summary  = tf.summary.scalar("loss/var_norm", self.var_norm)
        self.huber_loss_summary = tf.summary.scalar("loss/huber_loss", self.huber_loss)
        self.q_summaries = []
        for i in range(self.num_actions):
            self.q_summary = tf.summary.histogram("q_summary" + "/action_" + str(i), self.q_values[:, i])
            self.q_summaries.append(self.q_summary)

    def merge_summaries(self):
        self.summarize = tf.summary.merge([self.loss_summary,
                                           self.huber_loss_summary,
                                           self.grad_summary,
                                           self.var_summary] + self.q_summaries)

    def create_variables(self):
        self.create_input_placeholders()
        self.create_variables_for_q_values()
        self.create_variables_for_target()
        self.create_variables_for_optimization()
        self.create_variables_for_target_network_update()
        self.create_summaries()
        self.merge_summaries()

    def compute_all_q_values(self, states):
        return self.session.run(self.q_values, {self.states: states})

    def update_parameters(self, batch):
        train_itr = self.session.run(self.global_step)
        write_summary = train_itr % self.summary_every == 0
        _, summary = self.session.run([self.train_op,
                                       self.summarize if write_summary else self.no_op],
                                      {self.states: batch["states"],
                                       self.actions: batch["actions"],
                                       self.rewards: batch["rewards"],
                                       self.next_states: batch['next_states'],
                                       self.dones: batch["dones"]})

        self.session.run(self.target_update)

        if write_summary:
            self.summary_writer.add_summary(summary, train_itr)
