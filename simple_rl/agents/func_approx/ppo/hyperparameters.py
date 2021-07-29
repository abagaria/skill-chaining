ppo_params = {
    "lr": 3e-4,
    "update_interval": 2048,
    "batch_size": 64,
    "entropy_coef": 1e-3,
    "epochs": 4,
    "gamma": 0.995,
    "lambd": 0.97,
    "observation_norm": True,
    "eps": 1e-5
}

rnd_params = {
    "lr": 1e-4,
    "n_epochs": 1,
    "batch_size": 64,
    "update_interval": 2048,
}