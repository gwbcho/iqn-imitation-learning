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
    actions = tf.Variable(tf.zeros((num_segrots, action_dim)))
    actions[0].assign(segrot[0])
    actions[1:num_segrots - 1, :].assign(
        segrot[1:num_segrots - 1, :] - segrot[0:num_segrots - 2, :]
    )
    # important actions to reduce need for exploration
    # 3 5:6 8:9 18 20:21 30:31 33 35:36 45 47:48 57:58 62 66 73 75 78:80
    reduced_actions = tf.concat([
        tf.expand_dims(actions[:, 3], 1),
        actions[:, 5:7],
        actions[:, 8:10],
        tf.expand_dims(actions[:, 18], 1),
        actions[:, 20:22],
        actions[:, 30:32],
        tf.expand_dims(actions[:, 33], 1),
        tf.expand_dims(actions[:, 45], 1),
        actions[:, 47:49],
        actions[:, 57:59],
        tf.expand_dims(actions[:, 62], 1),
        tf.expand_dims(actions[:, 66], 1),
        tf.expand_dims(actions[:, 73], 1),
        tf.expand_dims(actions[:, 75], 1),
        actions[:, 78:81]
    ], 1)
    return reduced_actions
