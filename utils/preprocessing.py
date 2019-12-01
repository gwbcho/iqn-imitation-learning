import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def get_data(file='data/COS071212_mocap_processed.mat'):
    """
    Get expert states and actions returned as a tuple
    (training_states, training_actions, test_states, test_actions) where the states and actions
    are numpy arrays of shape (num_items, state_dimension) and (num_items, action_dimension)
    respectively.

    Args:
        file (String): string indicating the data file location

    Returns:
        Tuple containing tensorflow matricies (segrot, srates, markpos).
    """
    mat_data = loadmat(file)
    segrot = tf.Variable(mat_data['segrot'], dtype='float32')
    # srates are the smoothed neuron firing rates
    srates = tf.Variable(mat_data['srates'], dtype='float32')
    markpos = tf.Variable(mat_data['markpos'], dtype='float32')
    return segrot, srates, markpos


def get_actions_from_segrot(segrot):
    """
    Function to determine the the iterative actions from a segrots matrix

    Args:
        segrot (tf.Variable): (experiment_length, action_dim)

    Returns:
        Tensorflow variable containing actions (experiment_length - 1, action_dim)
    """
    num_segrots = segrot.shape[0]
    action_dim = segrot.shape[1]
    actions = tf.zeros((num_segrots, action_dim))
    actions[0] = (segrot[i])
    for i in range(1, num_segrots):
        actions[i] = (segrot[i] - segrot[i - 1])
    return actions
