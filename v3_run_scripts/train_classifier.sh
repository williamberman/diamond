#! /bin/bash

set -e
set -u

# game="$1"
# num_classes="$2"

game="Breakout"
num_classes=4

torchrun --nproc_per_node=1 src/train_classifier.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set_our_format/ \
    --save_dir /mnt/raid/diamond_v3/${game}/classifier_512k/ \
    --train_n_examples 512000 \
    --batch_size 512 \
    --validation_batch_size 1024 \
    --num_classes ${num_classes} \
    --num_input_images 30 \
    --validation_subset_n 10000 \
    --validation_steps 1000 \
    --save_steps 1000 \
    --wandb_name ${game}_512k
