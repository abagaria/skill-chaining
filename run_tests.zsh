# Variables
experiment_name='(cs2951x) dsc_opt_pes_nu_0.5'

mkdir -p $experiment_name

# Main loop
for num_run in {1..10}; do
	log_file="${experiment_name}/run_${num_run}.log"
	python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env="maze" --experiment_name=${experiment_name} --episodes=300 --steps=1000 --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=100 --device="cuda:0" --num_subgoal_hits=5 --nu=0.5 --num_run=$num_run --seed=$num_run |& tee -a $log_file
	
	# Check for errors
	if grep -q 'Error' $log_file; then 
		echo "num_run=${num_run}" | mail -s "FAILED: ${experiment_name}" matthew.slivinski@uconn.edu;
	else
		echo "num_run=${num_run}" | mail -s "COMPLETED: ${experiment_name}" matthew.slivinski@uconn.edu;
	fi
done

# python3 simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env="maze" --experiment_name="(cs2951x) dsc_opt_pes" --episodes=300 --steps=1000 --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=100 --device="cuda:0" --num_subgoal_hits=5 --nu=0.5 --num_run=6 && echo "num_run=6" | mail -s "(cs2951x) dsc_opt_pes COMPLETE" matthew.slivinski@uconn.edu