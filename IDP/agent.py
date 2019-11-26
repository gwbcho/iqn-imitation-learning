# import external dependencies
import numpy as np
import tensorflow as tf

# import local dependencies
import environment.environment as environment
from IDP.networks import StochasticActor, AutoRegressiveStochasticActor, Critic, Value
from IDP.helpers import ReplayBuffer, update, ActionSampler


"""
File Description:

Class: Immitation Distributional Policy (IDP) agent.
"""


class IDPAgent:
    """
    IDP agent.
    Action is alway from -1 to 1 in each dimension.
    Will not do normalization.
    """
    def __init__(self, args):
        """
        Agent class to generate a stochastic policy.

        Args:
            args (class):
                Attributes:
                    TODO
        """
        self.args = args
        self.action_dim = args.action_dim
        self.state_dim = args.state_dim
        if args.actor == 'IQN':
            self.actor = StochasticActor(args.state_dim, args.action_dim)
        elif args.actor == 'AIQN':
            self.actor = AutoRegressiveStochasticActor(args.state_dim, args.action_dim)
        self.action_sampler = ActionSampler(self.actor.action_dim)

    def train_one_step(self):
        """
        Execute one update for each of the networks. Note that if no positive advantage elements
        are returned the algorithm doesn't update the actor parameters.

        Args:
            None

        Returns:
            None
        """
        # transitions is sampled from replay buffer
        transitions = self.replay.sample_batch(self.args.batch_size)
        # transitions.s is a set of states sampled from replay buffer
        states, actions, advantages = self._sample_positive_advantage_actions(transitions.s)
        if advantages.shape[0]:
            self.actor.train(
                states,
                actions,
                advantages,
                self.args.mode,
                self.args.beta
            )
        update(self.target_actor, self.actor, self.args.tau)

    def _sample_positive_advantage_actions(self, states):
        """
        Sample from the target network and a uniform distribution.
        Then only keep the actions with positive advantage.
        Returning one action per state, if more needed, make states contain the
        same state multiple times.

        Args:
            states (tf.Variable): states of dimension (batch_size, state_dim)

        Returns:
            good_states (list): Set of positive advantage states (batch_size, sate_dim)
            good_actions (list): Set of positive advantage actions
            advantages (list[float]): set of positive advantage values (Q - V)
        """
        # tile states to be of dimension (batch_size * K, state_dim)
        tiled_states = tf.tile(states, [self.args.action_samples, 1])
        # Sample actions with noise for exploration
        target_actions = self.action_sampler.get_actions(self.target_actor, tiled_states)
        target_actions += tf.random.normal(target_actions.shape) * 0.01
        target_actions = tf.clip_by_value(target_actions, -1, 1)
        target_q = self.target_critics(tiled_states, target_actions)
        # Sample multiple actions both from the target policy and from a uniform distribution
        # over the action space. These will be used to determine the target distribution
        random_actions = tf.random.uniform(target_actions.shape, minval=-1.0, maxval=1.0)
        random_q = self.target_critics(tiled_states, random_actions)
        # create target actions vector, consistent of purely random actions and noisy actions
        # for the sake of exploration
        target_actions = tf.concat([target_actions, random_actions], 0)
        # compute Q and V values with dimensions (2 * batch_size * K, 1)
        q = tf.concat([target_q, random_q], 0)
        # determine the estimated value of a given state
        v = self.target_value(tiled_states)
        v = tf.concat([v, v], 0)
        # expand tiled states to allow for indexing later on
        tiled_states = tf.concat([tiled_states, tiled_states], 0)
        # remove unused dimensions
        q_squeezed = tf.squeeze(q)
        v_squeezed = tf.squeeze(v)
        # select s, a with positive advantage
        squeezed_indicies = tf.where(q_squeezed > v_squeezed)
        # collect all advantegeous states and actions
        good_states = tf.gather_nd(tiled_states, squeezed_indicies)
        good_actions = tf.gather_nd(target_actions, squeezed_indicies)
        # retrieve advantage values
        advantages = tf.gather_nd(q-v, squeezed_indicies)
        return good_states, good_actions, advantages

    def get_action(self, states):
        """
        Get a set of actions for a batch of states

        Args:
            states (tf.Variable): dimensions (TODO)

        Returns:
            sampled actions for the given state with dimension (batch_size, action_dim)
        """
        return self.action_sampler.get_actions(self.actor, states)
