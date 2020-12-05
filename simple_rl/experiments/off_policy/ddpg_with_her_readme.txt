I'm traning on d4rl ant maze, and I wanna fix one starting point
(lower left corner) and then target three points, upper right corner,
upper left corner, lower right corner...

To start, we can just pass in empty test-time goal pickles, to get
a pretrained solver to use for the rest of the experiment (say 3 pretrained)
solvers, one for each seed (0, 1, 2) and just make sure to keep the seeds
consistent at test time

I am going to call the pretrains

kiran_capstone_pretrain_0

Ran:
python3 simple_rl/experiments/off_policy/ddpg_with_her.py --experiment_name "kiran_capstone_pretrain_0" --env "d4rl-ant-maze" --seed 0 --num_pretrain_episodes 2000 --num_episodes 0 --num_steps 2000 --device "cuda:1" --test_time_start_states_pickle "state_pickles/empty_array.pkl" --test_time_goal_states_pickle "state_pickles/empty_array.pkl" --her_at_test_time

kiran_capstone_pretrain_1

Ran:
python3 simple_rl/experiments/off_policy/ddpg_with_her.py --experiment_name "kiran_capstone_pretrain_1" --env "d4rl-ant-maze" --seed 1 --num_pretrain_episodes 2000 --num_episodes 0 --num_steps 2000 --device "cuda:1" --test_time_start_states_pickle "state_pickles/empty_array.pkl" --test_time_goal_states_pickle "state_pickles/empty_array.pkl" --her_at_test_time


kiran_capstone_pretrain_2

Ran:
python3 simple_rl/experiments/off_policy/ddpg_with_her.py --experiment_name "kiran_capstone_pretrain_2" --env "d4rl-ant-maze" --seed 2 --num_pretrain_episodes 2000 --num_episodes 0 --num_steps 2000 --device "cuda:0" --test_time_start_states_pickle "state_pickles/empty_array.pkl" --test_time_goal_states_pickle "state_pickles/empty_array.pkl" --her_at_test_time
