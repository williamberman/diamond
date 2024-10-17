#!/bin/bash

set -e
set -u

# game=$1
# gpu=$2
# has_negative_rewards=$3
# num_classes=$4

game="Breakout"
gpu=0
has_negative_rewards=0
num_classes=4

torchrun --nproc_per_node 6 src/train_classifier.py \
    --num_classes ${num_classes} \
    --num_input_images 30 \
    --has_negative_rewards ${has_negative_rewards} \
    --classifying actions \
    --training_dataset_path /mnt/raid/diamond_v2/${game}/classifier_training_set/ \
    --holdout_dataset_path /mnt/raid/diamond_v2/${game}/classifier_holdout_set/ \
    --validation_steps 500 \
    --save_steps 500 \
    --save_dir /mnt/raid/diamond_v2/${game}/action_classifier/
