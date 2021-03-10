rainbow_params = {
    'num_steps': 3,                     # for n-step returns
    'replay_start_size': 2*(10**4),     # number of steps before updating q-functions
    'update_interval': 4,               # number of env steps for each gradient update
    'lr': 6.25e-5,                      # Adam learning rate
    'target_update_interval': 8000,     # number of env steps before updating target network
    'betasteps': int(1e7)               # number of env steps over which the beta param is annealed in PER
}

efficient_rainbow_params = {
    'num_steps': 20,
    'replay_start_size': 1600,
    'update_interval': 1,
    'lr': 1e-4,
    'target_update_interval': 2000,
    'betasteps': int(1e5)
}
