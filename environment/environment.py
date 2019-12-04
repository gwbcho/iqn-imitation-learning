import sys

import numpy as np
import tensorflow as tf


def distance_from_expert(agent_actions, expert_actions):
    """
    Determine the euclidean distance from the expert action. The negative value of the distance can
    be used as the agent reward (it is monotonically increasing as the agent learns to mimic the
    expert). The function assumes the state is implicitly known.

    Args:
        agent_actions (tf.Variable): tensor/matrix containing the agent's action(s). Note that if
            a matrix is fed into the function, the dimension should respect (batch_size, action_dim)
        expert_action (tf.Variable): tensor containing the expert action(s) of the same dimension as
            the agent actions

    Returns:
        distance values as a tensor with dimensions (batch_size, 1)
    """
    single_vector_used = False
    # account for the agent actions being a single vector
    if len(agent_actions.shape) == 1:
        single_vector_used = True
        agent_actions = tf.expand_dims(agent_actions, 0)
        expert_actions = tf.expand_dims(expert_actions, 0)
    diff = tf.math.squared_difference(agent_actions, expert_actions)
    sum_of_diff = tf.math.reduce_sum(diff, 1)  # sum over all action differences from expert
    euclidean_distance = tf.math.sqrt(sum_of_diff)
    return euclidean_distance


class IDPEnvironment(object):

    def __init__(self, expert_states, expert_actions, max_steps=1000):
        self.expert_states = expert_states
        self.expert_actions = expert_actions
        self.max_steps = max_steps

        self.final_index = self.expert_states.shape[0] - 1
        self.index = np.random.randint(0, self.final_index)
        self.current_step = 0
        self.prev_action = self.expert_actions[self.index]
        self.current_state = tf.concat([self.expert_states[self.index], self.prev_action], 0)

        self.is_terminal = False
        self.action_dim = self.expert_actions.shape[1]
        self.state_dim = self.expert_states.shape[1] + self.expert_actions.shape[1]

    def step(self, action):
        # convert actions [-1, 1] to [-180, 180]
        action = action * 180
        if self.is_terminal:
            return self.current_state, -sys.maxsize, self.is_terminal
        self.index += 1
        self.current_step += 1
        if self.index >= self.final_index or self.current_step >= self.max_steps:
            self.is_terminal = True
        self.prev_action = action
        new_state = self.expert_states[self.index]
        new_state = tf.concat([new_state, self.prev_action], 0)
        expert_state = tf.concat(
            [
                self.expert_states[self.index],
                self.expert_actions[self.index]
            ],
            0
        )
        expert_action = self.expert_actions[self.index]
        self.current_state = new_state
        rewards = -distance_from_expert(action, expert_action)[0]
        rewards += -distance_from_expert(new_state, expert_state)[0]
        return self.current_state, rewards, self.is_terminal

    def reset(self):
        rand_int = np.random.randint(0, self.final_index)
        self.is_terminal = False
        self.prev_action = self.expert_actions[0]
        self.current_state = tf.concat([self.expert_states[0], self.prev_action], 0)
        self.index = rand_int
        self.current_step = 0
        return self.current_state

    def sample_action(self):
        return tf.random.uniform([1, self.action_dim], minval=-1, maxval=1)
