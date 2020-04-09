# Variables
declare -a experiment_names=("(cs2951x) dsc_opt_pes_nu_0.8")
declare -a nus=(0.8)
declare -a seeds=(6)

num_runs=${#experiment_names[@]}

# Main loop
for i in {1..${num_runs}}; do	
	# Run variables
	experiment_name=${experiment_names[i]}
	nu=${nus[i]}
	seed=${seeds[i]}
	
	mkdir -p $experiment_name

	# Run and log script
	log_file="${experiment_name}/run_${seed}.log"
	python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env="maze" --experiment_name=$experiment_name --episodes=300 --steps=1000 --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=100 --device="cuda:0" --num_subgoal_hits=5 --nu=$nu --num_run=$seed --seed=$seed |& tee -a $log_file

	# Check for errors
	if grep -q -i 'Error' $log_file; then 
		echo "num_run=${seed}" | mail -s "FAILED: ${experiment_name}" matthew.slivinski@uconn.edu;
	else
		echo "num_run=${seed}" | mail -s "COMPLETED: ${experiment_name}" matthew.slivinski@uconn.edu;
	fi
done