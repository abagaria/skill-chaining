#$-cwd
#$-l long
#$-l gpus=1
#$-e chunked-lambda-investigation-error
#$-o chunked-lambda-investigation-output
#$-t 1-36
#$-tc 10
. /home/abagaria/.bashrc
. /home/abagaria/miniconda3/bin/activate base

# a good starting point is 0.5 * sqrt(3000) = 27. Because the old scaling term was on repulsive, while this determines attractive.

seed=$(($SGE_TASK_ID - 1))

echo seed $seed

### OLD CORRECT ###
# num_steps_array=(500 1000 2000 3000 4000 5000)
# # lambda_array=(11 16 22 27 31 35)
# lambda_array=(1 5 10 27 50 100) # Arbitrary but gives a good spread?
# epochs_array=(1800 1800 450 200 112 72)  # Just related to num_steps_array, to ensure roughly consistent gradient updates. Chunk size 1000.
# num_steps_index=$(($seed % 6))
# lambda_index=$(($seed / 6))
### END OLD CORRECT ###


##### Begin messing with things #####
# We're not going to collect 500, because below chunk size scaling will work differently.
# Be aware I've reorganized the lambdas and stuff!
# Compared to before, we need to have our lambdas be 31.6 times bigger. Because we used to be scaling BONUS down by that much.

# What we expect our "best" lambdas to be, to get a range around.
# | 1000 | 2000 | 3000 | 4000 | 5000 |
# | 29   | 8    | 3.8  | 2    | 1.6  |

# num_steps_array=(5000 5000 5000 2000 2000 2000 2000 3000 3000 3000 3000 3000 4000 4000 4000 4000) # 3x5, 4x2, 5x3,4x4.
# lambda_array=(0.02 0.03 0.04 0.125 0.15 0.2 0.35 0.03 0.05 0.07 0.12 0.18 0.025 0.035 0.06 0.075)
# epochs_array=(72 72 72 450 450 450 450 200 200 200 200 200 112 112 112 112)
num_steps_array=(1000 1000 1000 1000 1000 2000 2000 2000 2000 2000 3000 3000 3000 3000 3000 4000 4000 4000 4000 4000 5000 5000 5000 5000 5000)
lambda_array=(12 20 29 45 60 4 5.5 8 12 16 1.8 2.7 3.8 5.2 6.9 1.0 1.6 2.0 3.0 4.8 0.6 1.0 1.6 2.2 3.0)
epochs_array=(1800 1800 1800 1800 1800 450 450 450 450 450 200 200 200 200 200 112 112 112 112 112 72 72 72 72 72)
# num_steps_array=(2000 4000 5000 500 1000 500 1000 500 1000 2000 3000 4000 5000)
# lambda_array=(0.25 0.05 0.05 0.8 0.8 0.9 0.9 3 3 3 3 3 3) # Arbitrary but gives a good spread?
# epochs_array=(450 112 72 1800 1800 1800 1800 1800 1800 450 200 112 72)  # Just related to num_steps_array, to ensure roughly consistent gradient updates. Chunk size 1000.
num_steps_index=$seed
lambda_index=$seed

##### End messing with things #####


num_steps=${num_steps_array[$num_steps_index]}
lambda=${lambda_array[$lambda_index]}
epochs=${epochs_array[$num_steps_index]}

run_title=seed-$seed-steps-$num_steps-lambda-$lambda-epochs-$epochs

echo num_steps $num_steps
echo lambda $lambda
echo epochs $epochs
echo run-title $run_title

path_to_function=simple_rl/agents/func_approx/exploration/optimism/latent/experiments/experiment13.py # for legibility

python -u $path_to_function --experiment_name="writes/exp13-mcar-grid-scaling-bugfix" --run_title $run_title --seed $seed --steps $num_steps --lam $lambda --optimization_quantity chunked-bonus --bonus_scaling_term none --epochs $epochs --env_name "MountainCar-v0" --pixel_observation --lam_scaling_term none

#python -u simple_rl/agents/func_approx/exploration/optimism/latent/experiments/experiment12.py --experiment_name="exp12" --run_title DQN-Eps0 --seed $seed --episodes=100 --steps=200 --exploration_method="eps-const" --eval_eps=0.
