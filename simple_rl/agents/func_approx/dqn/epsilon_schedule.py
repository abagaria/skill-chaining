class EpsilonSchedule:
    def __init__(self, eps_start, eps_end, eps_exp_decay, eps_linear_decay_length):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_exp_decay = eps_exp_decay
        self.eps_linear_decay_length = eps_linear_decay_length
        self.eps_linear_decay = (eps_start - eps_end) / eps_linear_decay_length

    def update_epsilon(self, current_epsilon, num_executions):
        pass

class GlobalEpsilonSchedule(EpsilonSchedule):
    def __init__(self, eps_start, eps_end=0.05):
        EPS_END = eps_end
        EPS_EXPONENTIAL_DECAY = 0.999
        EPS_LINEAR_DECAY_LENGTH = 100000
        super(GlobalEpsilonSchedule, self).__init__(eps_start, EPS_END, EPS_EXPONENTIAL_DECAY, EPS_LINEAR_DECAY_LENGTH)

    def update_epsilon(self, current_epsilon, num_executions):
        if num_executions < self.eps_linear_decay_length:
            return current_epsilon - self.eps_linear_decay
        if num_executions == self.eps_linear_decay_length:
            print("Global Epsilon schedule switching to exponential decay")
        return max(self.eps_end, self.eps_exp_decay * current_epsilon)

class OptionEpsilonSchedule(EpsilonSchedule):
    def __init__(self, eps_start, eps_end=0.05):
        EPS_END = eps_end
        EPS_EXPONENTIAL_DECAY = 0.999
        EPS_LINEAR_DECAY_LENGTH = 10000
        super(OptionEpsilonSchedule, self).__init__(eps_start, EPS_END, EPS_EXPONENTIAL_DECAY, EPS_LINEAR_DECAY_LENGTH)

    def update_epsilon(self, current_epsilon, num_executions):
        return max(self.eps_end, self.eps_exp_decay * current_epsilon)

class ConstantEpsilonSchedule(EpsilonSchedule):
    def __init__(self, eps):
        self.eps = eps
        EPS_END = 0.05
        EPS_EXPONENTIAL_DECAY = 0.
        EPS_LINEAR_DECAY_LENGTH = 10000
        super(ConstantEpsilonSchedule, self).__init__(eps, EPS_END, EPS_EXPONENTIAL_DECAY, EPS_LINEAR_DECAY_LENGTH)

    def update_epsilon(self, current_epsilon, num_executions):
        return self.eps