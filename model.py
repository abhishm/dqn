import tensorflow as tf
import numpy as np

def q_network(states, observation_dim, num_actions):
    observation_dim = np.prod(observation_dim)
    x = states
    W1 = tf.get_variable("W1", [observation_dim, 20],
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [20],
                         initializer=tf.constant_initializer(0))
    z1 = tf.matmul(states, W1) + b1
    x = tf.nn.elu(z1)
    W2 = tf.get_variable("W2", [20, num_actions],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [num_actions],
                          initializer=tf.constant_initializer(0))
    q = tf.matmul(x, W2) + b2
    return q
