import tensorflow as tf
import numpy as np
import random
import tflearn
import os

class CoveringOptions(object):
    # This class identifies a subgoal by Laplacian method.
    # We feed this option to the skill chaining as a parent and generate its child options.
    
    def __init__(self, replay_buffer, obs_dim, threshold=0.95, num_units=200, num_training_steps=1000, actor_learning_rate=0.0001, critic_learning_rate=0.0001, batch_size=64, option_idx=None, name="covering-options"):
        self.obs_dim = obs_dim
        self.threshold = threshold
        self.num_units = num_units
        self.num_training_steps = num_training_steps
        self.batch_size = batch_size
        self.option_idx = option_idx
        self.name = name

        self.initiation_classifier = SpectrumNetwork(obs_dim=self.obs_dim, training_steps=self.num_training_steps, n_units=self.num_units, conv=False, name=self.name + "-spectrum")

        self.initiation_classifier.initialize()
        self.train(replay_buffer)
        
        self.threshold_value = self.sample_f_val(replay_buffer)

        
        # Parameters to be set to child options.
        # We don't use them for covering options.
        class mock():
            def __init__(self):
                self.actor_learning_rate = actor_learning_rate
                self.critic_learning_rate = critic_learning_rate
                self.batch_size = batch_size
                
        self.solver = mock()
        

    def train(self, replay_buffer):
        for _ in range(self.num_training_steps):
            s, a, r, s2, t = replay_buffer.sample(self.batch_size)
            
            next_f_value = self.initiation_classifier(s2)
            self.initiation_classifier.train(s, next_f_value)
            
        
    def is_init_true(self, ground_state):
        s = np.asarray([ground_state.data])
        # print('s=', s)
        # print('s=', type(s))
        return self.initiation_classifier(s) > self.threshold_value

    def is_term_true(self, ground_state):
        return True

    def sample_f_val(self, experience_buffer):
        buf_size = len(experience_buffer)

        n_samples = min(buf_size, 2048)
        # n_samples = buf_size

        # s = [experience_buffer.memory[i][0] for i in range(len(experience_buffer.memory))]
        s, _, _, _, _ = experience_buffer.sample(n_samples)
        f_values = self.initiation_classifier(s)
        if type(f_values) is list:
            f_values = np.asarray(f_values)
        # print('fvalue=', f_values)
        f_values = f_values.flatten()

        f_srt = np.sort(f_values)
        
        print('f_srt=', f_srt)

        # print('n_samples=', n_samples)
        # print('len(s)=', len(s))
        # print('f_value=', len(f_values))
        
        init_th = f_srt[int(n_samples * self.threshold)]

        print('init_th =', init_th)

        return init_th

    
class SpectrumNetwork():

    def __init__(self, obs_dim=None, learning_rate=0.001, training_steps=100, batch_size=32, n_units=16, beta=2.0, delta=0.1, feature=None, conv=False, name="spectrum"):
        # Beta  : Lagrange multiplier. Higher beta would make the vector more orthogonal.
        # delta : Orthogonality parameter.
        self.sess = tf.Session()
        self.learning_rate = learning_rate
        self.obs_dim = obs_dim

        self.n_units = n_units
        
        # self.beta = 1000000.0
        self.beta = beta
        self.delta = 0.05
        # self.delta = delta

        self.feature = feature

        self.conv = conv
        
        self.name = name

        self.obs, self.f_value = self.network(scope=name+"_eval")

        self.next_f_value = tf.placeholder(tf.float32, [None, 1], name=name+"_next_f")

        # TODO: Is this what we are looking for?
        self.loss = tflearn.mean_square(self.f_value, self.next_f_value) + \
                    self.beta * tf.reduce_mean(tf.multiply(self.f_value - self.delta, self.next_f_value - self.delta)) + \
                    self.beta * tf.reduce_mean(self.f_value * self.f_value * self.next_f_value * self.next_f_value) + \
                    self.beta * (self.f_value - self.next_f_value) # This is to let f(s) <= f(s').
        
        # with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
            
        self.optimize = self.optimizer.minimize(self.loss)

        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name + "_eval")
        self.initializer = tf.initializers.variables(self.network_params + self.optimizer.variables())

        # print('network param names for ', self.name)
        # for n in self.network_params:
        #     print(n.name)
            
        self.saver = tf.train.Saver(self.network_params)

    def network(self, scope):
        # TODO: What is the best NN?
        if self.feature is None:
            indim = self.obs_dim
        else:
            indim = self.feature.num_features()
            
        obs = tf.placeholder(tf.float32, [None, indim], name=self.name+"_obs")

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            if self.conv:
                reshaped_obs = tf.reshape(obs, [-1, 105, 80, 3])
                net = tflearn.conv_2d(reshaped_obs, 32, 8, strides=4, activation='relu')
                net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
                out = tflearn.fully_connected(net, 1, weights_init=tflearn.initializations.uniform(minval=-0.003, maxval=0.003))
            else:
                net = tflearn.fully_connected(obs, self.n_units, name='d1', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(indim)))
                net = tflearn.fully_connected(net, self.n_units, name='d2', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.n_units)))
                net = tflearn.fully_connected(net, self.n_units, name='d3', weights_init=tflearn.initializations.truncated_normal(stddev=1.0/float(self.n_units)))
                # net = tflearn.layers.normalization.batch_normalization(net)
                # net = tf.contrib.layers.batch_norm(net)
                net = tflearn.activations.relu(net)

                w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
                out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return obs, out

    def train(self, obs1, next_f_value):
        # obs1 is
        obs = []
        # print('obs1=', obs1)
        for state in obs1:
            # print('state=', state)
            # print('state.data=', state.data)
            o = state.data
            obs.append(o)

        # print('next_f_value=', next_f_value)
        # print('type(obs)=', type(obs))
        # print('type(next_f_value)=', type(next_f_value))
        self.sess.run(self.optimize, feed_dict={
            self.obs: obs,
            self.next_f_value: next_f_value
        })

    def initialize(self):
        self.sess.run(self.initializer, feed_dict={})

    def f_ret(self, state):
        return self.sess.run(self.f_value, feed_dict={
            self.obs: state
        })

    def f_from_features(self, features):
        assert(isinstance(features, np.ndarray))
        return self.sess.run(self.f_value, feed_dict={
            self.obs: features
        })
    
    def __call__(self, obs):
        assert(isinstance(obs, np.ndarray))        
        
        # assert(isinstance(obs1, list))        
        # obs = [state.data for state in obs1]
        # 
        # ret = []
        # for o in obs:
        #     print('o=', o)
        #     print('o=', o.shape)
        #     r = self.f_ret(o)
        #     print('r=', r)
        #     print('r=', r.shape)
        #     ret.append(r.flatten())
        # return ret
        
        r = self.f_ret(obs)
        return r

    def restore(self, directory, name='spectrum_nn'):
        self.saver.restore(self.sess, directory + '/' + name)
    
    def save(self, directory, name='spectrum_nn'):
        self.saver.save(self.sess, directory + '/' + name)

