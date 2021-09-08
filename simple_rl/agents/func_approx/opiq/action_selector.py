import numpy as np
import logging


class OptimisticAction:

    def __init__(self, count_model, epsilon_start, epsilon_finish,
                 epsilon_time_length, num_actions, optim_m, optim_action_tau):

        self.epsilon_start = epsilon_start
        self.epsilon_finish = epsilon_finish
        self.epsilon_time_length = epsilon_time_length

        self.num_actions = num_actions
        self.logger = logging.getLogger("OptimisticAction")

        self.m = optim_m
        self.tau = optim_action_tau
        self.count_model = count_model

    def select_actions(self, state, q_values, t, testing=False):

        epsilon = max(
            self.epsilon_finish,
            self.epsilon_start
            - (t / self.epsilon_time_length)
            * (self.epsilon_start - self.epsilon_finish),
            )

        if not testing:
            self._log_eps(epsilon, t)

        if not testing and np.random.random() < epsilon:
            action = np.random.randint(self.num_actions)
            self.logger.debug("Random action selected")
            return action

        q_vals_copy = (q_values + 0).detach()[0].cpu().numpy()
        state_action_counts = self.count_model.get_all_action_counts(state)[:, 0]
        optims = self.tau / ((state_action_counts + 1.0) ** self.m)

        optim_q_vals = q_vals_copy + optims
        return np.argmax(optim_q_vals)

    def _log_eps(self, epsilon, t):
        self.logger.debug("Epsilon: {:.2f}".format(epsilon))
        if t % 1000 == 0:
            self.logger.info("Epsilon: {:.2f}".format(epsilon))
