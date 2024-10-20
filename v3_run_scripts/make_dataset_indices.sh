#! /bin/bash

set -e
set -u

# game=$1

game="Breakout"

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 512000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_512k.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 256000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_256k.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 128000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_128k.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 64000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_64000.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 32000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_32000.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 16000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_16000.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 8000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_8000.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 4000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_4000.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 2000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_2000.json

python src/make_dataset_indices.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set \
    --frame_count 1000 \
    --output_path /mnt/raid/diamond_v3/${game}/training_set_indices_1000.json


