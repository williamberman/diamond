#! /bin/bash

ctr=0

while true; do
    export SAVE_PREFIX="/workspace/test_search_$ctr" 
    echo "Saving to $SAVE_PREFIX"
    torchrun --nproc_per_node=8 src/searchy_boi.py
    ctr=$((ctr+1))
done
