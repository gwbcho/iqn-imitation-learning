import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tqdm import trange

import utils.utils as utils
from agents.GAC.agent import GACAgent
from environment.environment import IDPEnvironment
from noises.ounoise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from noises.param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from utils.preprocessing import get_data, get_actions_from_segrot


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description='An implementation of the Distributional Policy Optimization paper.',
    )
    parser.add_argument(
        '--gamma', type=float, default=0.99, metavar='G',
        help='discount factor for reward (default: 0.99)'
    )
    parser.add_argument(
        '--tau', type=float, default=5e-3, metavar='G',
        help='discount factor for model (default: 0.005)'
    )
    parser.add_argument('--noise', default='normal', choices=['ou', 'param', 'normal'])
    parser.add_argument(
        '--noise_scale', type=float, default=0.2, metavar='G',
        help='(default: 0.2)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=64, metavar='N',
        help='batch size (default: 64)'
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
        '--training_steps', type=int, default=50, metavar='N',
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
        '--rand_init', default=False, action='store_true',
        help='Randomly initialize the starting positions for the expert environment. The default value is False.'
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
        '-f', '--expert_file', type=str, default='data/COS071212_mocap_processed.mat',
        help='file to find expert actions for algorithm'
    )
    parser.add_argument(
        '-c', '--curtail_length', type=int, default=None, help='max expert data length to consider.'
    )
    parser.add_argument(
        '-m', '--max_steps', type=int, default=1000, help='max environment steps before termination.'
    )
    parser.add_argument(
        '-v', '--verbose', default=False, action='store_true', help='Store more verbose reward information.'
    )
    parser.add_argument(
        '-ct', '--create_testset', default=False, action='store_true', help='Create a test set from the data provided.'
    )
    parser.add_argument(
        '-rf', '--results_file', default='results.txt', help='Results file name/location.'
    )
    return parser


def _reset_noise(agent, a_noise, p_noise):
    if a_noise is not None:
        a_noise.reset()
    if p_noise is not None:
        agent.perturb_actor_parameters(p_noise)


def evaluate_policy(policy, env, episodes):
    """
    Run the environment env using policy for episodes number of times.
    Return: average rewards per episode.
    """
    rewards = []
    for _ in range(episodes):
        state = np.float32(env.reset())
        is_terminal = False
        while not is_terminal:
            action = policy.get_action(tf.convert_to_tensor([state], dtype=tf.float32))
            # remove the batch_size dimension if batch_size == 1
            action = tf.squeeze(action, [0]).numpy()
            state, reward, is_terminal = env.step(action)
            state, reward = np.float32(state), np.float32(reward)
            rewards.append(float(reward))
            # env.render()
    return rewards


def main():
    print(tf.__version__)
    print("GPU Available: ", tf.test.is_gpu_available())

    args = create_argument_parser().parse_args()

    segrot, states, markpos = get_data(file=args.expert_file)
    actions = get_actions_from_segrot(segrot)

    if args.curtail_length:
        states = states[0:args.curtail_length + 1]
        actions = actions[0:args.curtail_length + 1]

    action_dim = actions.shape[1]
    state_dim = states.shape[1]
    args.action_dim = action_dim
    args.state_dim = state_dim + action_dim

    """
    Create environment
    """
    env = IDPEnvironment(states[1:], actions[1:], args.max_steps, args.rand_init)
    eval_env = IDPEnvironment(states[1:], actions[1:], args.max_steps, args.rand_init)

    if args.create_testset:
        num_states = states.shape[0]
        num_train = int(0.9 * num_states)
        num_test = num_states - num_train
        train_states = states[1:num_train]
        train_actions = actions[1:num_train]
        test_states = states[-num_test:]
        test_actions = actions[-num_test:]
        env = IDPEnvironment(train_states, train_actions, args.max_steps, args.rand_init)
        eval_env = IDPEnvironment(test_states, test_actions, args.max_steps, args.rand_init)

    if args.noise == 'ou':
        noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(args.action_dim),
            sigma=float(args.noise_scale) * np.ones(args.action_dim)
        )
    elif args.noise == 'normal':
        noise = NormalActionNoise(
            mu=np.zeros(args.action_dim),
            sigma=float(args.noise_scale) * np.ones(args.action_dim)
        )
    else:
        noise = None

    if args.noise == 'param':
        param_noise = AdaptiveParamNoiseSpec(
            initial_stddev=args.noise_scale,
            desired_action_stddev=args.noise_scale
        )
    else:
        param_noise = None

    base_dir = os.getcwd() + '/models/GACAgent/'
    run_number = 0
    while os.path.exists(base_dir + str(run_number)):
        run_number += 1
    base_dir = base_dir + str(run_number)
    os.makedirs(base_dir)

    gac = GACAgent(**args.__dict__)
    state = env.reset()
    results_dict = {
        'train_rewards': [],
        'eval_rewards': [],
        'actor_losses': [],
        'value_losses': [],
        'critic_losses': []
    }
    episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode

    num_steps = args.num_steps
    if num_steps is not None:
        nb_epochs = int(num_steps) // (args.epoch_cycles * args.rollout_steps)
    else:
        nb_epochs = 500

    _reset_noise(gac, noise, param_noise)
    """
    training loop
    """
    average_rewards = 0
    count = 0
    total_steps = 0
    train_steps = 0
    for epoch in trange(nb_epochs):
        for cycle in range(args.epoch_cycles):
            for rollout in range(args.rollout_steps):
                """
                Get an action from neural network and run it in the environment
                """
                # print('t:', t)
                if total_steps < args.start_timesteps:
                    action = env.sample_action()
                else:
                    action = gac.select_perturbed_action(
                        tf.convert_to_tensor([state], dtype=tf.float32),
                        noise,
                        param_noise
                    )
                # remove the batch_size dimension if batch_size == 1
                action = tf.squeeze(action, [0]).numpy()
                # modify action from [-1, 1] to [-180, 180]
                next_state, reward, is_terminal = env.step(action)
                next_state, reward = np.float32(next_state), np.float32(reward)
                gac.store_transition(state, action, reward, next_state, is_terminal)
                episode_rewards += reward
                # print('average_rewards:', average_rewards)

                # check if game is terminated to decide how to update state, episode_steps,
                # episode_rewards
                if is_terminal:
                    state = np.float32(env.reset())
                    results_dict['train_rewards'].append(
                        (total_steps, episode_rewards)
                    )
                    episode_steps = 0
                    episode_rewards = 0
                    _reset_noise(gac, noise, param_noise)
                else:
                    state = next_state
                    episode_steps += 1

                # evaluate
                if total_steps % args.eval_freq == 0:
                    eval_rewards = evaluate_policy(gac, eval_env, args.eval_episodes)
                    eval_reward = sum(eval_rewards) / args.eval_episodes
                    eval_variance = float(np.var(eval_rewards))
                    if args.verbose:
                        results_dict['eval_rewards'].append({
                            'total_steps': total_steps,
                            'train_steps': train_steps,
                            'average_eval_reward': eval_reward,
                            'eval_reward_variance': eval_variance,
                            'eval_rewards_list': eval_rewards
                        })
                    else:
                        results_dict['eval_rewards'].append({
                            'total_steps': total_steps,
                            'train_steps': train_steps,
                            'average_eval_reward': eval_reward,
                            'eval_reward_variance': eval_variance
                        })
                    with open(args.results_file, 'w') as file:
                        file.write(json.dumps(results_dict['eval_rewards']))

                total_steps += 1
            # train
            if gac.replay.size >= args.batch_size:
                for _ in range(args.training_steps):
                    if train_steps % args.param_noise_interval == 0 and param_noise is not None:
                        episode_transitions = gac.replay.sample_batch(args.batch_size)
                        states = episode_transitions.s
                        unperturbed_actions = gac.get_action(states)
                        perturbed_actions = episode_transitions.a
                        ddpg_dist = ddpg_distance_metric(
                            perturbed_actions.numpy(),
                            unperturbed_actions.numpy()
                        )
                        param_noise.adapt(ddpg_dist)

                    gac.train_one_step()
                    train_steps += 1

    with open('results.txt', 'w') as file:
        file.write(json.dumps(results_dict))

    utils.save_model(gac.actor, base_dir)


if __name__ == '__main__':
    main()
