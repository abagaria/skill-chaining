import ipdb
import numpy as np


class ReplayMemory:
    def __init__(self):
        self.total_state = []
        self.total_action = []
        self.total_policy = []
        self.total_reward = []
        self.total_done = []
        self.total_next_state = []
        self.total_int_reward = []
        self.total_next_obs = []
        self.total_ext_value = []
        self.total_int_value = []

    def store(self, state, action, policy, ext_reward, int_reward, ext_value, int_value, next_state, done):
        self.check_input_shapes(state, action, policy, ext_reward, int_reward, ext_value, int_value, next_state, done)

        self.total_state.append(state)
        self.total_action.append(action)
        self.total_policy.append(policy)
        self.total_reward.append(ext_reward)
        self.total_int_reward.append(int_reward)
        self.total_ext_value.append(ext_value)
        self.total_int_value.append(int_value)
        self.total_next_state.append(next_state)
        self.total_done.append(done)

        next_obs = next_state[-1, ...]
        assert next_state.shape == (4, 84, 84), next_state.shape
        assert next_obs.shape == (84, 84), next_obs.shape
        self.total_next_obs.append(next_obs)

    def retrieve(self):
        state = np.array(self.total_state)
        action = np.array(self.total_action)
        policy = self.total_policy
        reward = np.array(self.total_reward)
        int_reward = np.array(self.total_int_reward)
        ext_value = np.array(self.total_ext_value)
        int_value = np.array(self.total_int_value)
        next_state = np.array(self.total_next_state)
        next_obs = np.array(self.total_next_obs)
        done = np.array(self.total_done)

        data = state, action, policy, reward, int_reward, ext_value, int_value, next_state, next_obs, done
        self.check_output_shapes(*data)
        return data

    def clear(self):
        self.total_state = []
        self.total_action = []
        self.total_policy = []
        self.total_reward = []
        self.total_done = []
        self.total_next_state = []
        self.total_int_reward = []
        self.total_next_obs = []
        self.total_ext_value = []
        self.total_int_value = []

    def check_input_shapes(self, state, action, policy, ext_reward, int_reward, ext_value, int_value, next_state, done):
        assert state.shape == (4, 84, 84), state.shape
        assert _isscalar(action), action
        assert _isscalar(ext_reward), ext_reward
        assert _isscalar(int_reward), int_reward
        assert _isscalar(ext_value), type(ext_value)
        assert _isscalar(int_value), int_value
        assert next_state.shape == (4, 84, 84), next_state.shape
        assert _isscalar(done), done

    def check_output_shapes(self, state, action, policy, reward, int_reward, ext_value, int_value, next_state, next_obs, done):
        assert state.shape == (len(self), 4, 84, 84), state.shape
        assert action.shape == (len(self),), action.shape
        assert reward.shape == (len(self),), reward.shape
        assert int_reward.shape == (len(self),), int_reward.shape
        assert ext_value.shape == (len(self),), ext_value.shape
        assert int_value.shape == (len(self),), int_value.shape
        assert next_state.shape == (len(self), 4, 84, 84), next_state.shape
        assert next_obs.shape == (len(self), 84, 84), next_obs.shape
        assert done.shape == (len(self),), done.shape

    def __len__(self):
        return len(self.total_state)

def _isscalar(x):
    return np.isscalar(x) or isinstance(x, (float, int))
