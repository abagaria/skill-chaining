# Command to wait for test to finish to run another
# echo Waiting...; while ps -p $PID > /dev/null; do sleep 1; done; zsh run_tests.zsh

# Variables
# env="treasure_game-v0"

env="maze"
episodes=300
steps=2000
declare -a nus=(0.5 0.6 0.7 0.8)
declare -a starts=(1 6 1 1)
declare -a ends=(10 10 10 10)
num_runs=${#nus[@]}
hits=5

# Add extra arguments for the treasure domain
if [[ "${env}" = "treasure" ]]; then
	extra_args="--discrete_actions=True --episodic_plots=True"
else
	extra_args=""
fi

# Main loop
for i in {1..${num_runs}}; do
	# Run variables
	nu=${nus[i]}
	experiment_name="(cs2951x) ${env}_chain_break_nu_${nu}"
	start=${starts[i]}
	end=${ends[i]}

	exp_dir="runs/${experiment_name}"

	mkdir -p ${exp_dir}

	# Run loop
	for seed in {${start}..${end}}; do
		log_file="${exp_dir}/run_${seed}.log"
		
		# Log header
		echo "=====================================" |& tee -a ${log_file}
		echo "START: $(date)" |& tee -a ${log_file}
		echo "CMD: python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env=${env} --experiment_name=${experiment_name} --episodes=${episodes} --steps=${steps} --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=20 --device="cuda:0" --num_subgoal_hits=${hits} --nu=${nu} --num_run=${seed} --seed=${seed}" ${extra_args} |& tee -a ${log_file}
		echo "- experiment_name: ${experiment_name}" |& tee -a ${log_file}
		echo "- nu: ${nu}" |& tee -a ${log_file}
		echo "- seed: ${seed}" |& tee -a ${log_file}
		echo "=====================================" |& tee -a ${log_file}
		echo "" |& tee -a ${log_file}

		# Run script
		start=`date +%s`
		python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env=${env} --experiment_name=${experiment_name} --episodes=${episodes} --steps=${steps} --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=20 --device="cuda:0" --num_subgoal_hits=${hits} --nu=${nu} --num_run=${seed} --seed=${seed} ${extra_args} |& tee -a ${log_file}
		end=`date +%s`
		runtime=$((end-start))

		# Calculate runtime
		h=$(( runtime / 3600 ))
		m=$(( (runtime % 3600) / 60 ))
		s=$(( (runtime % 3600) % 60 ))

		# Log footer
		echo "" |& tee -a ${log_file}
		echo "=====================================" |& tee -a ${log_file}
		echo "END: $(date)" |& tee -a ${log_file}
		echo "RUNTIME: $h:$m:$s (h:m:s)" |& tee -a ${log_file}
		echo "=====================================" |& tee -a ${log_file}

		# Check for errors and email results
		if grep -q -i 'Error' ${log_file}; then 
			echo "seed=${seed}\nRUNTIME: $h:$m:$s (h:m:s)" | mail -s "FAILED: ${experiment_name}" matthew.slivinski@uconn.edu;
		else
			echo "seed=${seed}\nRUNTIME: $h:$m:$s (h:m:s)" | mail -s "COMPLETED: ${experiment_name}" matthew.slivinski@uconn.edu;
		fi
	done
done