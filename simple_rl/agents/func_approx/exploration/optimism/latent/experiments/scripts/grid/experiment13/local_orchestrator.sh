#!/bin/bash

# cd /home/sam/Code/ML/optimism/skill-chaining

for i in {1..36}
do
    export SGE_TASK_ID=$i
    ./chunked_lambda_investigation.sh
done
