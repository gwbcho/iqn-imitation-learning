# import external dependencies
import numpy as np
import tensorflow as tf

# import local dependencies
import environment.environment as environment
from agents.networks.networks import StochasticActor, AutoRegressiveStochasticActor, AIQNRNN, IQNFFN
from agents.helpers import ReplayBuffer, update, ActionSampler


"""
File Description:

Class: Immitation Distributional Policy (IDP) agent.
"""


class IDPAgent:
    """
    GAC agent.
    Action is always from -1 to 1 in each dimension.
    Will not do normalization.
    """
    def __init__(self, action_dim, state_dim, states, expert_actions, action_samples=10,
                 mode='linear', beta=1, batch_size=64, actor='AIQN', expert_noise=0.01,
                 *args, **kwargs):
        """
        Agent class to generate a stochastic policy.

        Args:
            action_dim (int): action dimension
            state_dim (int): state dimension
            states (tf.Variable): states for the entire arm motion
            expert_actions (tf.Variable): expert actions for the correspoding states (minus the
                terminal state)
            action_samples (int): originally labelled K in the paper, represents how many
                actions should be sampled from the memory buffer
            mode (string): poorly named variable to represent variable being used in the
                distribution being used
            beta (float): value used in boltzmann distribution
            batch_size (int): batch size
            q_normalization (float): q value normalization rate
            actor (string): string indicating the type of actor to use
            expert_noise (float): expert noise to regularize expert actions
        """
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.action_samples = action_samples
        self.mode = mode
        self.beta = beta
        self.batch_size = batch_size
        self.expert_noise = expert_noise

        # expert states and actions
        self.states = states
        self.expert_actions = expert_actions

        # determine how many states and actions there are
        self.state_len = states.shape[0]
        self.expert_actions_len = expert_actions.shape[0]
        self.state_indicies = range(self.state_len)
        self.expert_action_indicies = range(self.expert_actions_len)

        # type of actor being used
        self.actor = actor
        if self.actor == 'IQN':
            self.actor = StochasticActor(self.state_dim, self.action_dim)
        elif self.actor == 'AIQN':
            self.actor = AutoRegressiveStochasticActor(self.state_dim, self.action_dim)
        elif self.actor == 'RNN':
            self.actor = AIQNRNN(self.state_dim, self.action_dim)
        elif self.actor == 'FFN':
            self.actor = IQNFNN(self.state_dim, self.action_dim)

    def train_actor(self, state_batch, expert_action_batch):
        """
        Execute one update for each of the networks. Note that if no positive advantage elements
        are returned the algorithm doesn't update the actor parameters.

        Args:
            state_batch (tf.Variable): states determined from the expert actor (as states are
                independent of actions in actuality) (batch_size, state_dim)
            expert_action_batch (tf.Variable): expert actions to compare the agent to
                (batch_size, action_dim)

        Returns:
            None
        """
        if self.actor == 'RNN' or self.actor == 'FFN':
            # Train the RNN or FFN actor using supervised methods.
            with tf.GradientTape() as tape:
                predicted_actions = self.actor(state_batch)
                loss = self.actor.loss(expert_action_batch, predicted_actions)

            gradients = tape.gradient(loss, self.actor.trainable_variables)
            self.actor.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables))
            return

        # TODO: select a random set of actions
        # determine a set of positive advantage states, actions, and advantages from the "target"
        # network which, for our purposes, is simply the expert action with gaussian noise for
        # normalization purposes
        good_states, good_actions, advantages = self._sample_positive_advantage_actions(
            state_batch,
            expert_action_batch
        )
        if advantages.shape[0]:
            self.actor.train(
                good_states,
                good_actions,
                advantages,
                self.args.mode,
                self.args.beta
            )

    def _sample_positive_advantage_actions(self, states, expert_actions, expert_noise=0.01):
        """
        Sample from the target network and a uniform distribution.
        Then only keep the actions with positive advantage.
        Returning one action per state, if more needed, make states contain the
        same state multiple times.

        Args:
            states (tf.Variable): states of dimension (batch_size, state_dim)
            expert_actions (tf.Variable): expert actions of dimension (batch_size, action_dim)
            expert_noise (float): variable indicating expert noise to allow for regularization and
                to act as target "distribution" for IQN

        Returns:
            good_states (list): Set of positive advantage states (batch_size, sate_dim)
            good_actions (list): Set of positive advantage actions
            advantages (list[float]): set of positive advantage values this is intuitively how good
                a given action is given the state. For our purposes this can be considered the
                euclidean distance from the expert actions
        """
        # tile states to be of dimension (batch_size * K, state_dim)
        tiled_states = tf.tile(states, [self.action_samples, 1])
        tiled_expert_actions = tf.tile(expert_actions, [self.action_samples, 1])
        # Sample actions with noise for exploration
        target_actions = (
            tiled_expert_actions + tf.random.normal(tiled_expert_actions.shape) * expert_noise
        )
        target_actions = tf.clip_by_value(target_actions, -1, 1)
        advantages = -environment.distance_from_expert(tiled_expert_actions, target_actions)
        return tiled_states, target_actions, advantages

    def get_action(self, states):
        """
        Get a set of actions for a batch of states

        Args:
            states (tf.Variable): dimensions (TODO)

        Returns:
            sampled actions for the given state with dimension (batch_size, action_dim)
        """
        return self.action_sampler.get_sampled_actions(self.actor, states)
