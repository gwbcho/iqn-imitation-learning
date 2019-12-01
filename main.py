import os
import argparse

import gym
import numpy as np
import tensorflow as tf

from agents.IDP.agent import IDPAgent
from agents.GAC.agent import GACAgent
from utils.preprocessing import get_data, get_actions_from_segrot


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description='An implementation of the Distributional Policy Optimization paper.',
    )
    parser.add_argument(
        '--environment', default="HalfCheetah-v2",
        help='name of the environment to run. default="HalfCheetah-v2"'
    )
    parser.add_argument(
        '--gamma', type=float, default=0.99, metavar='G',
        help='discount factor for reward (default: 0.99)'
    )
    parser.add_argument(
        '--tau', type=float, default=5e-3, metavar='G',
        help='discount factor for model (default: 0.005)'
    )
    parser.add_argument('--noise', default='ou', choices=['ou', 'param', 'normal'])
    parser.add_argument(
        '--noise_scale', type=float, default=0.2, metavar='G',
        help='(default: 0.2)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, metavar='N',
        help='batch size (default: 64)'
    )
    parser.add_argument(
        '--epochs', type=int, default=1000, metavar='N',
        help='number of training epochs (default: None)'
    )
    parser.add_argument(
        '--epoch_cycles', type=int, default=20, metavar='N',
        help='default=20'
    )
    parser.add_argument(
        '--rollout_steps', type=int, default=100, metavar='N',
        help='default=100'
    )
    parser.add_argument(
        '--T', type=int, default=50, metavar='N',
        help='number of training steps (default: 50)'
    )
    parser.add_argument(
        '--model_path', type=str, default='/tmp/dpo/',
        help='trained model is saved to this location, default="/tmp/dpo/"'
    )
    parser.add_argument('--param_noise_interval', type=int, default=50, metavar='N')
    parser.add_argument('--start_timesteps', type=int, default=10000, metavar='N')
    parser.add_argument('--eval_freq', type=int, default=5000, metavar='N')
    parser.add_argument('--eval_episodes', type=int, default=10, metavar='N')
    parser.add_argument(
        '--buffer_size', type=int, default=1000000, metavar='N',
        help='size of replay buffer (default: 1000000)'
    )
    parser.add_argument('--action_samples', type=int, default=16)
    parser.add_argument('--visualize', default=False, action='store_true')
    parser.add_argument(
        '--experiment_name', default=None, type=str,
        help='For multiple different experiments, provide an informative experiment name'
    )
    parser.add_argument('--print', default=False, action='store_true')
    parser.add_argument('--actor', default='IQN', choices=['IQN', 'AIQN', 'RNN', 'FFN'])
    parser.add_argument(
        '--normalize_obs', default=False, action='store_true', help='Normalize observations'
    )
    parser.add_argument(
        '--normalize_rewards', default=False, action='store_true', help='Normalize rewards'
    )
    parser.add_argument(
        '--q_normalization', type=float, default=0.01,
        help='Uniformly smooth the Q function in this range.'
    )
    parser.add_argument(
        '--mode', type=str, default='linear', choices=['linear', 'max', 'boltzmann', 'uniform'],
        help='Target policy is constructed based on this operator. default="linear" '
    )
    parser.add_argument(
        '--beta', type=float, default=1.0,
        help='Boltzmann Temperature for normalizing actions, default=1.0'
    )
    parser.add_argument(
        '--num_steps', type=int, default=2000000, metavar='N',
        help='number of training steps to play the environments game (default: 2000000)'
    )
    parser.add_argument(
        '--expert_noise', type=float, default=0.01, help='noise for expert actions.'
    )
    return parser


def random_preprocessing(inputs, labels):
    """
    Preprocess the data further by shuffling inputs and labels randomly

    Args:
        inputs (tf.Variable): variable containing a 2d vector of inputs
        labels (tf.Variable): variable containing a 1d vector of labels (predicted words)

    Returns:
        shuffled inputs and corresponding labels
    """
    indices = range(0, labels.shape[0])
    shuffled_indices = tf.random.shuffle(indices)
    inputs = tf.gather(inputs, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)
    return inputs, labels


def train(model, train_states, train_actions, batch_size):
    """
    Function to train the inputted model on the provided training states and actions

    Args:
        model (IDPAgent): An IDPAgent instance to train on the expert strategy
        train_states (tf.Variable): states to be used for training
        train_actions (tf.Variable): expert actions for training

    Returns:
        None
    """
    # randomly shuffile input data to increase accuracy
    shuffled_states, shuffled_actions = random_preprocessing(train_states, train_actions)

    tl = train_states.shape[0]  # training inputs length
    mod = tl % batch_size  # number of batches resulting from batch size
    N = int(np.floor(tl/batch_size))  # number for splitting training data
    split_details = [batch_size] * N  # N elements of batch_size [batch_size, batch_size, ...]
    split_details.append(mod)
    shuffled_states = tf.split(shuffled_states, split_details)
    shuffled_actions = tf.split(shuffled_actions, split_details)

    for j in range(tl):
        # Implement backprop:
        model.train_actor(shuffled_states, shuffled_actions)


def main():
    print(tf.__version__)
    print("GPU Available: ", tf.test.is_gpu_available())

    args = create_argument_parser().parse_args()

    segrot, srates, markpos = get_data(file='data/COS071212_mocap_preprocessed.mat')
    actions = get_actions_from_segrot(segrot)
    action_dim = actions.shape[1]
    state_dim = states.shape[1]
    args.action_dim = action_dim
    args.state_dim = state_dim

    base_dir = os.getcwd() + '/model/'
    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    os.makedirs(base_dir)

    idp_agent = IDPAgent(**args.__dict__)
    for _ in range(args.epochs):
        train(idp_agent, srates, actions, args.batch_size)

    # TODO: train the network

    utils.save_model(idp_agent.actor, base_dir)


if __name__ == '__main__':
    main()
