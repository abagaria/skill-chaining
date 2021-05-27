import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6), device=torch.device("cuda")):
		self.max_size = max_size
		self.state_dim = state_dim
		self.action_dim = action_dim

		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.intrinsic_reward = np.zeros((max_size, 1))
		self.extrinsic_reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))

		self.device = device

	def add(self, state, action, intrinsic_reward, extrinsic_reward, next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.intrinsic_reward[self.ptr] = intrinsic_reward
		self.extrinsic_reward[self.ptr] = extrinsic_reward
		self.next_state[self.ptr] = next_state
		self.done[self.ptr] = done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.intrinsic_reward[ind]).to(self.device),
			torch.FloatTensor(self.extrinsic_reward[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)

	def __len__(self):
		return self.size

	def __getitem__(self, i):
		if i < self.size:
			return self.state[i], self.action[i], self.intrinsic_reward[i], self.extrinsic_reward[i], self.next_state[i], self.done[i]
		raise IndexError(f"Tried to access index {i} when length is {self.size}")

	def clear(self):
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((self.max_size, self.state_dim))
		self.action = np.zeros((self.max_size, self.action_dim))
		self.next_state = np.zeros((self.max_size, self.state_dim))
		self.intrinsic_reward = np.zeros((self.max_size, 1))
		self.extrinsic_reward = np.zeros((self.max_size, 1))
		self.done = np.zeros((self.max_size, 1))
