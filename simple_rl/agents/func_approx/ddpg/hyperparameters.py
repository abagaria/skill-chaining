# Hyperparameters
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.01
LRA = 1e-4
LRC = 1e-4
HIDDEN_1 = 400
HIDDEN_2 = 300

MAX_EPISODES = 50000
MAX_STEPS = 200
GLOBAL_LINEAR_EPS_DECAY = 1e-5  # complete decay over 1 hundred thousand transitions
OPTION_LINEAR_EPS_DECAY = 2e-5  # complete decay over 50 thousand transitions
EPS_MIN                 = 0.25  # always add some noise to prevent getting stuck
PRINT_EVERY = 10
