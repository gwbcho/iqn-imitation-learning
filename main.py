import os
import argparse

import gym
import numpy as np
import tensorflow as tf

from IDP.networks import AutoRegressiveStochasticActor as AIQN
from IDP.networks import StochasticActor as IQN
from IDP.networks import Critic, Value
from IDP.agent import IDPAgent


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
        '--epochs', type=int, default=None, metavar='N',
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
    parser.add_argument('--actor', default='IQN', choices=['IQN', 'AIQN'])
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
        '--mode', type=str, default='linear', choices=['linear', 'max', 'boltzman', 'uniform'],
        help='Target policy is constructed based on this operator. default="linear" '
    )
    parser.add_argument(
        '--beta', type=float, default=1.0,
        help='Boltzman Temperature for normalizing actions, default=1.0'
    )
    parser.add_argument(
        '--num_steps', type=int, default=2000000, metavar='N',
        help='number of training steps to play the environments game (default: 2000000)'
    )
    return parser


def main():
    print(tf.__version__)
    print("GPU Available: ", tf.test.is_gpu_available())

    args = create_argument_parser().parse_args()

    action_dim = 38
    state_dim = 100
    args.action_dim = action_dim
    args.state_dim = state_dim

    base_dir = os.getcwd() + '/model/'
    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    os.makedirs(base_dir)

    gac = IDPAgent(args)



    utils.save_model(gac.actor, base_dir)


if __name__ == '__main__':
    main()
