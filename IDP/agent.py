# import external dependencies
import numpy as np
import tensorflow as tf

# import local dependencies
from IDP.networks import StochasticActor, AutoRegressiveStochasticActor,  Critic, Value
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
        # transitions is sampled from replay buffer
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
        return critic_history, value_history

    def _sample_positive_advantage_actions(self, states):
        """
        Sample all more advantageous actions dtermined by the expert given the agent's current state


        Args:
            states (tf.Variable): dimension (batch_size * K, state_dim)

        Returns:
            good_states (list): Set of positive advantage states (batch_size, sate_dim)
            good_actions (list): Set of positive advantage actions
            advantages (list[float]): set of positive advantage values (Q - V)
        """
        # Sample actions
        actions = self.action_sampler.get_actions(self.target_actor, states)
        actions = tf.concat([actions, tf.random.uniform(actions.shape, minval=-1.0, maxval=1.0)], 0)
        states = tf.concat([states, states], 0)

        # compute Q and V dimensions (2 * batch_size * K, 1)
        q = self.critics(states, actions)
        v = self.value(states)
        # remove unused dimensions
        q_squeezed = tf.squeeze(q)
        v_squeezed = tf.squeeze(v)

        # select s, a with positive advantage
        squeezed_indicies = tf.where(q_squeezed > v_squeezed)
        # collect all advantegeous states and actions
        good_states = tf.gather_nd(states, squeezed_indicies)
        good_actions = tf.gather_nd(actions, squeezed_indicies)
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