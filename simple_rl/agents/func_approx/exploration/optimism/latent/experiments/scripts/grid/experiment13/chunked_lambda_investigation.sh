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

num_steps_array=(500 1000 2000 3000 4000 5000)
# lambda_array=(11 16 22 27 31 35)
lambda_array=(1 5 10 27 50 100) # Arbitrary but gives a good spread?

epochs_array=(1800 1800 450 200 112 72)  # Just related to num_steps_array, to ensure roughly consistent gradient updates. Chunk size 1000.

num_steps_index=$(($seed % 6))
lambda_index=$(($seed / 6))

num_steps=${num_steps_array[$num_steps_index]}
lambda=${lambda_array[$lambda_index]}
epochs=${epochs_array[$num_steps_index]}

run_title=seed-$seed-steps-$num_steps-lambda-$lambda-epochs-$epochs

echo num_steps $num_steps
echo lambda $lambda
echo epochs $epochs
echo run-title $run_title

path_to_function=simple_rl/agents/func_approx/exploration/optimism/latent/experiments/experiment13.py # for legibility

python -u $path_to_function --experiment_name="exp13-mcar-grid" --run_title $run_title --seed $seed --steps $num_steps --lam $lambda --optimization_quantity chunked-bonus --bonus_scaling_term none --epochs $epochs --env_name "MountainCar-v0" --pixel_observation

#python -u simple_rl/agents/func_approx/exploration/optimism/latent/experiments/experiment12.py --experiment_name="exp12" --run_title DQN-Eps0 --seed $seed --episodes=100 --steps=200 --exploration_method="eps-const" --eval_eps=0.
