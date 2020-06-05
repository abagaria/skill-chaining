import tensorflow as tf
import numpy as np
import random
import tflearn
import os
from numpy.linalg import norm
from tqdm import tqdm
from copy import deepcopy
import torch
import ipdb

from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.agents.func_approx.dqn.DQNAgentClass import ReplayBuffer
from simple_rl.mdp.StateClass import State

class CoveringOptions(Option):
    # This class identifies a subgoal by Laplacian method.
    # We feed this option to the skill chaining as a parent and generate its child options.

    def __init__(self, *, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size,
				 replay_buffer, obs_dim, feature=None, threshold=0.95, num_units=200, num_training_steps=1000,
                 batch_size=64,
                 option_idx=None,
                 subgoal_reward=0., max_steps=20000, seed=0,
                 dense_reward=False,
                 timeout=200,
                 device=torch.device("cpu"),
                 chain_id=1,
                 beta=0.0,
                 use_xy_prior=False):

        self.obs_dim = obs_dim
        self.threshold = threshold
        self.num_units = num_units
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.chain_id = chain_id
        self.option_idx = option_idx
        self.name = name
        self.beta = beta
        self.use_xy_prior = use_xy_prior

        # SkillChainingClass expects all options to have these attributes
        self.children = []
        self.max_num_children = 1
        self.initialize_everywhere = False

        if feature == "fourier":
            self.feature = Fourier(obs_dim, (-np.ones(obs_dim), np.ones(obs_dim)), 3)
        else:
            self.feature = None

        if self.feature is None:
            indim = self.obs_dim
        else:
            indim = self.feature.num_features()

        Option.__init__(self, overall_mdp, name, global_solver, lr_actor, lr_critic, ddpg_batch_size,
                        subgoal_reward=subgoal_reward, max_steps=max_steps, seed=seed,
                        dense_reward=dense_reward, enable_timeout=True,
                        timeout=timeout, option_idx=option_idx, device=device)

        self.initiation_classifier = SpectrumNetwork(obs_dim=indim, training_steps=self.num_training_steps,
                                                     n_units=self.num_units, conv=False,
                                                     name=self.name + "-spectrum", beta=beta)
        self.initiation_classifier.initialize()

        if self.use_xy_prior:
            replay_buffer = self.convert_states(replay_buffer)

        self.train(replay_buffer)

        self.threshold_value = self.sample_f_val(replay_buffer)

    def states_to_tensor(self, states):
        obs = []
        for state in states:
            if self.feature is None:
                # debugging the NN
                # o = np.zeros_like(state.data)
                # o[0] = state.data[0]
                # o[1] = state.data[1]
                o = state.data
            else:
                o = self.feature.feature(state, 0)
            obs.append(o)
        return obs

    def convert_states(self, replay_buffer):
        # This function converts all the states recorded in the replay buffer's memory to only have x and y
        new_rb = ReplayBuffer(action_size=2,
                              buffer_size=replay_buffer.buffer_size,
                              batch_size=64,
                              seed=0,
                              device=self.device)
        for old_exp in replay_buffer.memory:
            state, action, reward, next_state, terminal = old_exp.state, old_exp.action, old_exp.reward, old_exp.next_state, old_exp.done
            new_state = np.copy(state)
            new_state[2:] = 0
            new_next_state = np.copy(next_state)
            new_next_state[2:] = 0
            new_rb.add(new_state, action, reward, new_next_state, terminal, 1)
        return new_rb

    def train(self, replay_buffer):
        for _ in range(self.num_training_steps):
            s, a, r, s2, t, _ = replay_buffer.sample(min(self.batch_size, len(replay_buffer)))

            if not isinstance(s, np.ndarray):
                s = s.cpu().numpy()
                s2 = s2.cpu().numpy()

            # obs = list(obs)
            # print('s2=', s2)
            # obs2 = list(obs2)
            # print('obs2=', obs2)
            # s = list(s)
            # s2 = list(s2)
            obs2 = self.states_to_tensor(s2)

            next_f_value = self.initiation_classifier(obs2)

            obs = self.states_to_tensor(s)

            self.initiation_classifier.train(obs, next_f_value)

    def is_init_true(self, ground_state):
        s = self.states_to_tensor([ground_state])
        # print('s=', s)
        # print('s=', type(s))
        return self.initiation_classifier(s) > self.threshold_value

    def batched_is_init_true(self, state_matrix):
        x = self.initiation_classifier(state_matrix) > self.threshold_value
        return x.flatten()

    def get_chunked_initiation_probabilities(self, state_list, chunk_size=2000):  # TODO

        feature_list = self.states_to_tensor(state_list)
        num_states = len(feature_list)

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(num_states / chunk_size))
        state_chunks = np.array_split(feature_list, num_chunks, axis=0)

        initiation_probabilities = np.zeros((num_states, 1))
        current_idx = 0

        for chunk_number, state_chunk in tqdm(enumerate(state_chunks), desc="Chunked init"):  # type: int, np.ndarray
            chunk_values = self.initiation_classifier(state_chunk)
            current_chunk_size = len(state_chunk)
            initiation_probabilities[current_idx:current_idx + current_chunk_size] = chunk_values
            current_idx += current_chunk_size

        return initiation_probabilities

    def is_term_true(self, ground_state):
        return not self.is_init_true(ground_state)

    def sample_f_val(self, experience_buffer):
        buf_size = len(experience_buffer)

        n_samples = min(buf_size, 2048)
        # n_samples = buf_size

        # s = [experience_buffer.memory[i][0] for i in range(len(experience_buffer.memory))]
        transition = experience_buffer.sample(n_samples)
        s = transition[0]
        if not isinstance(s, np.ndarray):
            s = s.cpu().numpy()
        obs = self.states_to_tensor(s)
        f_values = self.initiation_classifier(obs)
        if type(f_values) is list:
            f_values = np.asarray(f_values)
        # print('fvalue=', f_values)
        f_values = f_values.flatten()

        f_srt = np.sort(f_values)

        # print('f_srt=', f_srt)

        # print('n_samples=', n_samples)
        # print('len(s)=', len(s))
        # print('f_value=', len(f_values))

        init_th = f_srt[int(n_samples * self.threshold)]

        print('init_th =', init_th)

        return init_th

    def _get_epsilon_greedy_epsilon(self):
        if "point" in self.overall_mdp.env_name:
            return 0.1
        elif "ant" in self.overall_mdp.env_name:
            return 0.25

    def act(self, state, eval_mode, warmup_phase):
        if random.random() < self._get_epsilon_greedy_epsilon() and not eval_mode:
            return self.overall_mdp.sample_random_action()
        return self.solver.act(state.features(), evaluation_mode=eval_mode)

    def update_option_solver(self, s, a, r, s_prime):
        """ Make on-policy updates to the current option's low-level DDPG solver. """
        assert not s.is_terminal(), "Terminal state did not terminate at some point"

        if self.is_term_true(s):
            print("[update_option_solver] Warning: called updater on {} term states: {}".format(self.name, s))
            return

        is_terminal = self.is_term_true(s_prime) or s_prime.is_terminal()
        subgoal_reward = self.get_dco_subgoal_reward(s, s_prime) if self.name != "global_option" else r
        self.solver.step(s.features(), a, subgoal_reward, s_prime.features(), is_terminal)

    def get_dco_subgoal_reward(self, state, next_state):
        s = state.features()[None, ...]
        sp = next_state.features()[None, ...]
        difference = self.initiation_classifier(s) - self.initiation_classifier(sp)
        if difference.shape != (1, 1):
            ipdb.set_trace()
        return difference[0][0]

    def execute_option_in_mdp(self, mdp, episode, step_number, eval_mode=False):
        """
        Option main control loop.

        Args:
            mdp (MDP): environment where actions are being taken
            episode (int)
            step_number (int): how many steps have already elapsed in the outer control loop.
            eval_mode (bool): Added for the SkillGraphPlanning agent so that we can perform
                              option policy rollouts with eval_epsilon but still keep training
                              option policies based on new experiences seen during test time.

        Returns:
            option_transitions (list): list of (s, a, r, s') tuples
            discounted_reward (float): cumulative discounted reward obtained by executing the option
        """
        start_state = deepcopy(mdp.cur_state)
        state = mdp.cur_state

        if self.is_init_true(state):
            option_transitions = []
            total_reward = 0.
            self.num_executions += 1
            num_steps = 0
            visited_states = []

            if self.name != "global_option":
                print("Executing {}".format(self.name))

            while not self.is_term_true(state) and not state.is_terminal() and \
                    step_number < self.max_steps and num_steps < self.timeout:

                action = self.act(state, eval_mode=eval_mode, warmup_phase=False)  # TODO
                reward, next_state = mdp.execute_agent_action(action, option_idx=self.option_idx)

                # This will update the global-solver when it is being executed
                self.update_option_solver(state, action, reward, next_state)

                if self.name != "global_option" and self.update_global_solver:
                    self.global_solver.step(state.features(), action, reward, next_state.features(),
                                            next_state.is_terminal())
                    self.global_solver.update_epsilon()

                option_transitions.append((state, action, reward, next_state))
                visited_states.append(state)

                total_reward += reward
                state = next_state

                # step_number is to check if we exhaust the episodic step budget
                # num_steps is to appropriately discount the rewards during option execution (and check for timeouts)
                step_number += 1
                num_steps += 1

            # Don't forget to add the final state to the followed trajectory
            visited_states.append(state)

            if self.is_term_true(state):
                print(f"{self} execution was successful")

            return option_transitions, total_reward


class SpectrumNetwork():

    def __init__(self, obs_dim=None, learning_rate=0.0001, training_steps=100, batch_size=32, n_units=16, beta=0.0,
                 delta=0.1, conv=False, name="spectrum"):
        # Beta  : Lagrange multiplier. Higher beta would make the vector more orthogonal.
        # delta : Orthogonality parameter.
        self.sess = tf.Session()
        self.learning_rate = learning_rate
        self.obs_dim = obs_dim

        self.n_units = n_units

        self.beta = beta
        self.delta = 0.05
        # self.delta = delta

        self.conv = conv

        self.name = name

        self.obs, self.f_value = self.network(scope=name + "_eval")

        self.next_f_value = tf.placeholder(tf.float32, [None, 1], name=name + "_next_f")

        # TODO: Is this what we are looking for?
        self.loss = tflearn.mean_square(self.f_value, self.next_f_value) \
                    + self.beta * tf.reduce_mean(tf.multiply(self.f_value - self.delta, self.next_f_value - self.delta)) \
                    + self.beta * tf.reduce_mean(self.f_value * self.f_value * self.next_f_value * self.next_f_value) \
                    + self.beta * tf.math.maximum((self.f_value - self.next_f_value),
                                                  0.0)  # This is to enforce f(s) <= f(s').

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.optimize = self.optimizer.minimize(self.loss)

        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "_eval")
        self.initializer = tf.initializers.variables(self.network_params + self.optimizer.variables())

        # print('network param names for ', self.name)
        # for n in self.network_params:
        #     print(n.name)

        self.saver = tf.train.Saver(self.network_params)

    def network(self, scope):
        indim = self.obs_dim

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.conv:
                obs = tf.placeholder(tf.float32, [None, 4, 84, 84], name=self.name + "_obs")
                # obs = tf.placeholder(tf.float32, [None, 105, 80, 3], name=self.name+"_obs")
                net = tflearn.conv_2d(obs, 32, 8, strides=4, activation='relu')
                net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
                out = tflearn.fully_connected(net, 1,
                                              weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
            else:
                obs = tf.placeholder(tf.float32, [None, indim], name=self.name + "_obs")
                # net = tflearn.fully_connected(obs, self.n_units, name='d1', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(indim)))
                # net = tflearn.fully_connected(net, self.n_units, name='d2', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.n_units)))
                # net = tflearn.fully_connected(net, self.n_units, name='d3', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.n_units)))
                # net = tflearn.layers.normalization.batch_normalization(net)
                # net = tf.contrib.layers.batch_norm(net)
                # net = tflearn.activations.relu(net)

                w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
                net = tflearn.fully_connected(obs, 1, weights_init=w_init)

                # @Yuu Jinnai: Do we not need any non-linearities here?
                #net = tflearn.fully_connected(obs, 3, weights_init=w_init, activation="relu")
                #net = tflearn.fully_connected(net, 1, weights_init=w_init)
                out = net
        return obs, out

    def train(self, obs, next_f_value):

        # print('next_f_value=', next_f_value)
        # print('type(obs)=', type(obs))
        # print('type(next_f_value)=', type(next_f_value))

        obs = [np.asarray(s) for s in obs]

        self.sess.run(self.optimize, feed_dict={
            self.obs: obs,
            self.next_f_value: next_f_value
        })

    def initialize(self):
        self.sess.run(self.initializer, feed_dict={})

    def f_ret(self, state):
        if isinstance(state, list):
            obs = [np.asarray(s) for s in state]
        else:
            assert isinstance(state, np.ndarray)
            obs = state

        return self.sess.run(self.f_value, feed_dict={
            self.obs: obs
        })

    def f_from_features(self, features):
        assert (isinstance(features, np.ndarray))
        return self.sess.run(self.f_value, feed_dict={
            self.obs: features
        })

    def __call__(self, obs):
        r = self.f_ret(obs)
        return r

    def restore(self, directory, name='spectrum_nn'):
        self.saver.restore(self.sess, directory + '/' + name)

    def save(self, directory, name='spectrum_nn'):
        self.saver.save(self.sess, directory + '/' + name)


class Fourier(object):
    def __init__(self, state_dim, bound, order):
        assert (type(state_dim) is int)
        assert (type(order) is int)

        assert (state_dim == bound[0].shape[0])
        assert (state_dim == bound[1].shape[0])

        self.state_dim = state_dim
        self.state_up_bound = bound[0]
        self.state_low_bound = bound[1]
        self.order = order

        self.coeff = np.indices((self.order,) * self.state_dim).reshape((self.state_dim, -1)).T

        n = np.array(list(map(norm, self.coeff)))
        n[0] = 1.0
        self.norm = 1.0 / n

    def feature(self, state, action):
        xf = state.data
        assert (len(xf) == self.state_dim)

        norm_state = (xf + self.state_low_bound) / (self.state_up_bound - self.state_low_bound)

        f_np = np.cos(np.pi * np.dot(self.coeff, norm_state))

        # Check if the weights are set to numbers
        assert (not np.isnan(np.sum(f_np)))

        return f_np.tolist()

    def alpha(self):
        return self.norm

    def num_features(self):
        # What is the number of features for Fourier?
        return self.order ** (self.state_dim)
