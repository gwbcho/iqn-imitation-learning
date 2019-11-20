import os
import argparse

import gym
import numpy as np
import tensorflow as tf

from IDP.networks import AutoRegressiveStochasticActor as AIQN
from IDP.networks import StochasticActor as IQN
from IDP.networks import Critic, Value
from IDP.agent import IDPAgent


def evaluate_policy(policy, env, episodes):
    """
    Run the environment env using policy for episodes number of times.
    Return: average rewards per episode.
    """
    total_reward = 0.0
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = policy.get_action(tf.convert_to_tensor([state]))
            # remove the batch_size dimension if batch_size == 1
            action = tf.squeeze(action, [0])
            state, reward, is_terminal, _ = env.step(action)
            total_reward += reward
            if is_terminal:
                break
    return total_reward / episodes


def main():
    args = create_argument_parser().parse_args()

    """
    Create Mujoco environment
    """
    env = gym.make(args.environment)
    args.action_dim = env.action_space.shape[0]
    args.state_dim = env.observation_space.shape[0]

    gac = GACAgent(args)

    state = env.reset()

    results_dict = {
        'train_rewards': [],
        'eval_rewards': [],
        'actor_losses': [],
        'value_losses': [],
        'critic_losses': []
    }
    episode_steps, episode_rewards = 0, 0 # total steps and rewards for each episode
    """
    training loop
    """
    average_rewards = 0
    count = 0
    for t in range(args.T):
        """
        Get an action from neural network and run it in the environment
        """
        print('t =', t)
        action = gac.get_action(tf.convert_to_tensor([state]))
        # remove the batch_size dimension if batch_size == 1
        action = tf.squeeze(action, [0])
        next_state, reward, is_terminal, _ = env.step(action)
        gac.store_transitions(state, action, reward, next_state, is_terminal)
        average_rewards = average_rewards + ((reward - average_rewards)/(count + 1))
        count += 1
        print('average_rewards:', average_rewards)

        # check if game is terminated to decide how to update state, episode_steps, episode_rewards
        if is_terminal:
            state = env.reset()
            results_dict['train_rewards'].append((t, episode_rewards / episode_steps))
            episode_steps = 0
            episode_rewards = 0
        else:
            state = next_state
            episode_steps += 1
            episode_rewards += reward

        # TODO add rollout
        # train
        if gac.replay.size >= args.batch_size:
            gac.train_one_step()

        # evaluate
        if t % args.eval_freq == 0:
            eval_reward = evaluate_policy(gac, env, args.eval_episodes)
            print('eval_reward:', eval_reward)
            results_dict['eval_rewards'].append((t, eval_reward))

if __name__ == '__main__':
    main()
