import unittest

import tensorflow as tf

import agents.networks.networks as networks


class TestGacNetworks(unittest.TestCase):

    def test_cosine_basis_linear(self):
        n_basis_functions = 64
        out_size = 400
        embedding = networks.CosineBasisLinear(n_basis_functions, out_size)
        batch_size_1 = 10
        batch_size_2 = 10
        # random input
        x = tf.Variable(
            tf.random.normal(
                [batch_size_1, batch_size_2],
                stddev=.1,
                dtype=tf.float32
            )
        )
        # simply test if this is working
        out = embedding(x)
        self.assertEqual(out.shape[0], batch_size_1)
        self.assertEqual(out.shape[1], batch_size_2)
        self.assertEqual(out.shape[2], out_size)

    def test_autoregressive_stochastic_actor_no_action(self):
        batch_size_1 = 1
        num_inputs = 10
        action_dim = 5
        n_basis_functions = 64
        # construct the autoregressive stochasitc actor for testing
        actor = networks.AutoRegressiveStochasticActor(
            num_inputs,
            action_dim,
            n_basis_functions
        )
        # taus and actions are column vectors
        state = tf.Variable(
            tf.random.normal(
                [batch_size_1, num_inputs],
                stddev=.1,
                dtype=tf.float32
            )
        )
        taus = tf.Variable(
            tf.random.uniform(
                [batch_size_1, action_dim, 1]
            )
        )
        action = actor(state, taus)
        self.assertEqual(action.shape[0], batch_size_1)
        self.assertEqual(action.shape[1], action_dim)

    def test_autoregressive_stochastic_actor_with_action(self):
        batch_size_1 = 10
        num_inputs = 20
        action_dim = 5
        n_basis_functions = 64
        # construct the autoregressive stochasitc actor for testing
        actor = networks.AutoRegressiveStochasticActor(
            num_inputs,
            action_dim,
            n_basis_functions
        )
        state = tf.Variable(
            tf.random.normal(
                [batch_size_1, num_inputs],
                stddev=.1,
                dtype=tf.float32
            )
        )
        # taus and actions are column vectors
        taus = tf.Variable(
            tf.random.uniform(
                [batch_size_1, action_dim, 1]
            )
        )
        prev_action = tf.Variable(
            tf.random.uniform(
                [batch_size_1, action_dim, 1]
            )
        )
        action = actor(state, taus, prev_action)
        self.assertEqual(action.shape[0], batch_size_1)
        self.assertEqual(action.shape[1], action_dim)

    def test_stochastic_actor(self):
        batch_size_1 = 10
        num_inputs = 20
        action_dim = 5
        n_basis_functions = 64
        # construct the autoregressive stochasitc actor for testing
        actor = networks.StochasticActor(
            num_inputs,
            action_dim,
            n_basis_functions
        )
        state = tf.Variable(
            tf.random.normal(
                [batch_size_1, num_inputs],
                stddev=.1,
                dtype=tf.float32
            )
        )
        # taus and actions are column vectors
        taus = tf.Variable(
            tf.random.uniform(
                [batch_size_1, action_dim, 1]
            )
        )
        prev_action = tf.Variable(
            tf.random.uniform(
                [batch_size_1, action_dim, 1]
            )
        )
        action = actor(state, taus, prev_action)
        self.assertEqual(action.shape[0], batch_size_1)
        self.assertEqual(action.shape[1], action_dim)


if __name__ == '__main__':
    unittest.main()
