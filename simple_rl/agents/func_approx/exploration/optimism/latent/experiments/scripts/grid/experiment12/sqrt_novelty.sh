 #$-cwd
 #$-l long
 #$-l gpus=1
 #$-e dqn-sqrt-novelty-error
 #$-o dqn-sqrt-novelty-output
 #$-t 1-10
 #$-tc 10
 . /home/abagaria/.bashrc
 . /home/abagaria/miniconda3/bin/activate base
 python -u simple_rl/agents/func_approx/exploration/optimism/latent/experiments/experiment12.py --experiment_name="exp12" --run_title DQN-Sqrt-Novelty --seed $SGE_TASK_ID --episodes=100 --steps=200 --exploration_method="count-phi" --use_bonus_during_action_selection=True --eval_eps=0. --bonus_scaling_term="sqrt"
