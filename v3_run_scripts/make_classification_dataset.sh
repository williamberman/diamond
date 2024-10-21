#! /bin/bash

set -e
set -u

# game="$1"

game="Breakout"

python src/make_classification_dataset.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set/ \
    --output_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set_our_format/ \
    --num_input_images 30 \
    --seed 42
