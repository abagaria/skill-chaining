# Command to wait for test to finish to run another
# echo Waiting...; while ps -p $PID > /dev/null; do sleep 1; done; zsh run_tests.zsh

# Variables
declare -a experiment_names=("(cs2951x) dsc_opt_pes_nu_0.5" "(cs2951x) dsc_opt_pes_nu_0.7")
declare -a nus=(0.5 0.7)
declare -a starts=(6 21)
declare -a ends=(10 25)
num_runs=${#experiment_names[@]}

# Main loop
for i in {1..${num_runs}}; do
	# Run variables
	experiment_name=${experiment_names[i]}
	nu=${nus[i]}
	start=${starts[i]}
	end=${ends[i]}

	mkdir -p ${experiment_name}

	# Run loop
	for seed in {${start}..${end}}; do
		log_file="${experiment_name}/run_${seed}.log"
		
		# Log header
		echo "=====================================" |& tee -a ${log_file}
		echo "START: $(date)" |& tee -a ${log_file}
		echo "CMD: python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env='maze' --experiment_name=${experiment_name} --episodes=300 --steps=1000 --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=20 --device='cuda:0' --num_subgoal_hits=5 --nu=${nu} --num_run=${seed} --seed=${seed}" |& tee -a ${log_file}
		echo "- experiment_name: ${experiment_name}" |& tee -a ${log_file}
		echo "- nu: ${nu}" |& tee -a ${log_file} |& tee -a ${log_file}
		echo "- seed: ${seed}" |& tee -a ${log_file} |& tee -a ${log_file}
		echo "=====================================" |& tee -a ${log_file}
		echo "" |& tee -a ${log_file}

		# Run script
		start=`date +%s`
		python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env="maze" --experiment_name=${experiment_name} --episodes=300 --steps=1000 --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=20 --device="cuda:0" --num_subgoal_hits=5 --nu=${nu} --num_run=${seed} --seed=${seed} |& tee -a ${log_file}
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
			echo "seed=${seed}" | mail -s "FAILED: ${experiment_name}" matthew.slivinski@uconn.edu;
		else
			echo "seed=${seed}" | mail -s "COMPLETED: ${experiment_name}" matthew.slivinski@uconn.edu;
		fi
	done
done