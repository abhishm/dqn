# DQN 
Modular implementation of DQN algorithm.

# Dependencies
* Python 2.7 or 3.5
* [TensorFlow](https://www.tensorflow.org/) 1.10
* [gym](https://pypi.python.org/pypi/gym) 
* [numpy](https://pypi.python.org/pypi/numpy)
* [tqdm](https://pypi.python.org/pypi/tqdm) progress-bar

# Features
- Using a neural network based as the function approximator for Q-learning
- Using a target network and soft-update to synchronoze target network with Q-network
- Using gradient clipping to make small but consistent updates towards optimal Q-network 

### Bonus
- Implementation of [Tabular Q-learnin](https://github.com/abhishm/dqn/tree/master/tabular_q_learning) 

# Usage

To train a model for Cartpole-v0:

	$ python test_graph_dqn.py 

To view the tensorboard

	$tensorboard --logdir .

# Results

- Tensorboard Progress Bar
![](https://github.com/abhishm/blog/blob/gh-pages/assets/images/2017-07-17-DQN/performance_cartpole.JPG)


