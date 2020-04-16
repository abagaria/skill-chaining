#!/usr/bin/env bash

threshold=$1
PLOT_DIR="threshold_${threshold}_plots"

mkdir -p $PLOT_DIR

for i in {0..5}; do
  python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py --env="point-maze" --experiment_name="sc_opt_pes_test" --episodes=20 --steps=2000 --use_smdp_update=True --option_timeout=True --subgoal_reward=300. --buffer_len=20 --device="cuda" --num_subgoal_hits=3 --threshold=$threshold
  mv "initiation_set_plots/sc_opt_pes_test/covering-options-0_covering-options-${threshold}_threshold.png" "${PLOT_DIR}/covering_options_${threshold}_${i}.png"
done
