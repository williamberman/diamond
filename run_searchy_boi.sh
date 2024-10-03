#! /bin/bash

ctr=0

while true; do
    export N_STEPS_AT_ONCE=$(($ctr % 5 + 1))
    export SAVE_PREFIX="/workspace/test_search_${ctr}_${N_STEPS_AT_ONCE}" 
    echo "Saving to $SAVE_PREFIX"
    torchrun --nproc_per_node=8 src/searchy_boi.py
    ctr=$((ctr+1))
done
