import ipdb
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import RMSprop
from simple_rl.agents.func_approx.opiq.model import DQN
from simple_rl.agents.func_approx.opiq.count import AtariCount
from simple_rl.agents.func_approx.opiq.replay_buffer import NStepReplayBuffer
from simple_rl.agents.func_approx.opiq.action_selector import OptimisticAction

# TODO: Does the count really depend on the full frame stack??

class OptimisticDQN:
    def __init__(self, input_shape, num_actions, device,
                 count_representation_shape,
                 eps_start, eps_finish, eps_length,
                 optim_m, optim_beta, optim_action_tau, optim_bootstrap_tau,
                 lr, max_grad_norm,
                 update_interval, batch_size, max_buffer_len, n_steps,
                 gamma):
        self.gamma = gamma
        self.device = device
        self.n_steps = n_steps
        self.optim_m = optim_m
        self.optim_beta = optim_beta
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.max_grad_norm = max_grad_norm
        self.update_interval = update_interval
        self.optim_action_tau = optim_action_tau
        self.optim_bootstrap_tau = optim_bootstrap_tau

        self.count_model = AtariCount(num_actions, count_representation_shape)
        self.action_selector = OptimisticAction(self.count_model, eps_start, eps_finish,
                                                eps_length, num_actions, optim_m, optim_action_tau)

        self.policy_network = DQN(input_shape, num_actions).to(device)
        self.target_network = DQN(input_shape, num_actions).to(device)

        self.optimizer = RMSprop(self.policy_network.parameters(), lr=lr, alpha=0.95, eps=0.00001, centered=True)
        self.replay_buffer = NStepReplayBuffer(gamma, max_buffer_len, n_steps, device)

        self.sync_target_network()

        self.num_updates = 0
        self.num_action_selections = 0

    def pre_process_image(self, obs):
        obs = np.array(obs)
        assert obs.dtype == np.uint8, obs.dtype
        obs = torch.as_tensor(obs).to(self.device).float() / 255.
        return obs.unsqueeze(0) if obs.shape == self.input_shape else obs

    @torch.no_grad()
    def act(self, state, eval_mode=False):
        self.num_action_selections += 1
        state = self.pre_process_image(state)
        q_values = self.policy_network(state)
        action = self.action_selector.select_actions(state, q_values, self.num_action_selections, eval_mode)
        return action

    @torch.no_grad()
    def visit(self, state, action):
        state = self.pre_process_image(state)
        self.count_model.visit(state, action)

    def update(self, state, action, reward, next_state, done):
        self.num_updates += 1
        self.replay_buffer.append(state, action, reward, next_state, done)

        if self.num_updates % self.update_interval == 0 and len(self.replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample(self.batch_size)
            self._learn(batch)

    def _learn(self, batch):
        state, action, reward, next_state, discount, done = batch["state"], batch["action"], batch["reward"],\
                                                            batch["next_state"], batch["discount"], batch["is_state_terminal"]
        n_states, n_actions, n_steps = batch["n_states"], batch["n_actions"], batch["n_steps"]

        Q_expected = self.policy_network(state).gather(1, action.unsqueeze(1))

        with torch.no_grad():
            rewards = self.compute_optimistic_n_step_target(n_states, n_actions, reward, n_steps)
            Q_targets_next = self.compute_optimistic_bootstrap_target(next_state)
            Q_targets = rewards + (discount * Q_targets_next * (1 - done))

        loss = F.mse_loss(Q_expected, Q_targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=self.max_grad_norm)
        self.optimizer.step()

    @torch.no_grad()
    def compute_optimistic_bootstrap_target(self, next_states):
        assert isinstance(next_states, torch.Tensor)

        next_q_values = self.target_network(next_states)
        next_state_bonuses = self.get_state_action_bonuses(next_states)

        assert next_q_values.shape == (next_states.shape[0], self.num_actions), next_q_values.shape
        assert next_state_bonuses.shape == (next_states.shape[0], self.num_actions), next_state_bonuses.shape

        augmented_state_action_values = next_q_values + next_state_bonuses
        assert augmented_state_action_values.shape == (next_states.shape[0], self.num_actions)

        # Indexing because torch.max also returns indices
        Q_targets_next = augmented_state_action_values.max(dim=1)[0]
        assert Q_targets_next.shape == (next_states.shape[0],), Q_targets_next.shape

        return Q_targets_next

    @torch.no_grad()
    def compute_optimistic_n_step_target(self, n_states, n_actions, n_rewards, n_steps):
        state_dim = (4, 84, 84)
        assert n_states.shape == (self.batch_size, self.n_steps, *state_dim), n_states.shape
        assert n_actions.shape == (self.batch_size, self.n_steps), n_actions.shape
        assert n_rewards.shape == (self.batch_size,), n_rewards.shape
        assert n_steps.shape == (self.batch_size,), n_steps.shape

        s_a_bonuses = self.get_state_action_bonuses(n_states.reshape(-1, *state_dim), n_actions.reshape(-1,))
        s_a_bonuses = s_a_bonuses.reshape(self.batch_size, self.n_steps)

        gamma_tensor = self._get_gamma_tensor(n_steps, self.gamma)
        discounted_n_step_returns = gamma_tensor * s_a_bonuses  # Element wise multiplication
        assert s_a_bonuses.shape == gamma_tensor.shape == discounted_n_step_returns.shape

        summed_discounted_n_step_returns = discounted_n_step_returns.sum(dim=1)
        assert summed_discounted_n_step_returns.shape == (self.batch_size,)

        return n_rewards + summed_discounted_n_step_returns

    def _get_gamma_tensor(self, n_steps, gamma):
        assert n_steps.shape == (self.batch_size,), n_steps.shape

        gamma_tensor = torch.zeros(self.batch_size, self.n_steps).to(self.device).float()

        for i in range(self.batch_size):
            n = n_steps[i]
            gamma_vector = [(gamma ** j) for j in range(n)]
            gamma_tensor[i, :n] = torch.as_tensor(gamma_vector).to(self.device).float()

        return gamma_tensor

    @torch.no_grad()
    def get_state_action_bonuses(self, states, actions=None):
        def bootstrap_counts_to_bonus(counts):
            return self.optim_bootstrap_tau / (counts + 1.).pow(self.optim_m)

        def s_a_counts_to_bonus(counts):
            return self.optim_beta / torch.sqrt(counts)

        s_a_counts = self.count_model.get_all_action_counts(states).transpose(1, 0)
        s_a_counts = torch.as_tensor(s_a_counts).to(self.device).float()

        if actions is None:
            return bootstrap_counts_to_bonus(s_a_counts)

        s_a_counts = s_a_counts.gather(dim=1, index=actions.unsqueeze(1))
        return s_a_counts_to_bonus(s_a_counts).clamp(-1, 1)

    @torch.no_grad()
    def sync_target_network(self):
        self.target_network.load_state_dict(self.policy_network.state_dict())
