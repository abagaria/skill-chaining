# Command to wait for a test to finish to run another
# echo Waiting...; while ps -p $PID > /dev/null; do sleep 1; done; zsh run_tests.zsh

# Variables
# env="treasure_game-v0"
env="maze"
episodes=300
steps=2000
declare -a pes_nus=(0.3)
declare -a opt_nus=(0.1)
declare -a starts=(6)
declare -a ends=(10)
num_runs=${#pes_nus[@]}
hits=5

# Add extra arguments for the treasure domain
if [[ "${env}" = "treasure" ]]; then
	extra_args='--discrete_actions=True --episodic_plots=True'
else
	# extra_args='--use_chain_fix=True'
	extra_args='--use_old=True'
	# extra_args=''
fi

# Main loop
for i in {1..${num_runs}}; do
	# Run variables
	pes_nu=${pes_nus[i]}
	opt_nu=${opt_nus[i]}
	experiment_name="(cs2951x) maze_old_continuous_learning_2"
	# experiment_name="(cs2951x) ${env}_pes_nu_${pes_nu}_with_chainfix"
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
		echo "CMD: python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env=${env} --experiment_name=${experiment_name} --episodes=${episodes} --steps=${steps} --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=20 --device="cuda:0" --num_subgoal_hits=${hits} --pes_nu=${pes_nu} --opt_nu=${opt_nu} --num_run=${seed} --seed=${seed} --episodic_saves=True" ${extra_args} |& tee -a ${log_file}
		echo "- experiment_name: ${experiment_name}" |& tee -a ${log_file}
		echo "- pes_nu: ${pes_nu}" |& tee -a ${log_file}
		echo "- opt_nu ${opt_nu}" |& tee -a ${log_file}
		echo "- seed: ${seed}" |& tee -a ${log_file}
		echo "=====================================" |& tee -a ${log_file}
		echo "" |& tee -a ${log_file}

		# Run script
		start=`date +%s`
		python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env=${env} --experiment_name=${experiment_name} --episodes=${episodes} --steps=${steps} --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=20 --device="cuda:0" --num_subgoal_hits=${hits} --pes_nu=${pes_nu} --opt_nu=${opt_nu} --num_run=${seed} --seed=${seed} --episodic_saves=True ${extra_args} |& tee -a ${log_file}
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