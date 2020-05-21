#!/usr/bin/env bash

threshold=${1:-0.1}
num_runs=${2:-5}
exp_name=${3:-dco_test}

SRC_DIR="initiation_set_plots/${exp_name}/"
DST_DIR="threshold_${threshold}_plots/"

mkdir -p "$DST_DIR"

for ((i = 0; i < num_runs; i++)); do
  python3 -u simple_rl/agents/func_approx/dsc/SkillChainingAgentClass.py \
    --experiment_name="$exp_name" \
    --device="cuda" \
    --env="point-maze" \
    --episodes=20 \
    --steps=2000 \
    --subgoal_reward=300.0 \
    --option_timeout=True \
    --num_subgoal_hits=3 \
    --buffer_len=20 \
    --use_smdp_update=True \
    --threshold="$threshold" \
    || exit

  mv "${SRC_DIR}"/covering-options*"${threshold}"_threshold*.png "${DST_DIR}" || exit
done
