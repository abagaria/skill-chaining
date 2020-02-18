#$-cwd
#$-l long
#$-l gpus=1
#$-e dqn-eps0-mcar-error
#$-o dqn-eps0-mcar-output
#$-t 1-10
#$-tc 10
. /home/abagaria/.bashrc
. /home/abagaria/miniconda3/bin/activate base
python -u simple_rl/agents/func_approx/exploration/optimism/latent/experiments/experiment12.py --experiment_name="exp12" --run_title DQN-Eps0 --seed $SGE_TASK_ID --episodes=100 --steps=200 --exploration_method="eps-const" --eval_eps=0.
