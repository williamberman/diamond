#! /bin/bash

ctr=0

while true; do
    export N_STEPS_AT_ONCE=$(($ctr % 5 + 1))
    export N_STEPS_AT_ONCE=$((6 - $N_STEPS_AT_ONCE))
    export SAVE_PREFIX="/workspace/test_search_backprop_${ctr}_${N_STEPS_AT_ONCE}" 
    echo "Saving to $SAVE_PREFIX"
    torchrun --nproc_per_node=8 src/searchy_boi.py
    ctr=$((ctr+1))
done
