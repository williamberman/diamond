#!/bin/bash

run_job() {
    local GPU=$1
    echo "Starting job on GPU $GPU"

    while true; do
        if python save_trajectories.py --save_dir /mnt/raid/orca_rl/trajectory_samples_2 --num_steps 12800000 --gpu $GPU; then
            echo "Job completed successfully on GPU $GPU"
            return 0
        else
            echo "Job failed on GPU $GPU. Retrying immediately..."
        fi
    done
}

# Run jobs on GPUs 0-7
for GPU in {0..7}; do
    run_job $GPU &
done

# Wait for all background jobs to finish
wait