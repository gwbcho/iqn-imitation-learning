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
