#! /bin/bash

set -e
set -u

# game="$1"
# num_classes="$2"
# model="$3"

game="Breakout"
num_classes=4
model="resnet_sweep_2"

torchrun --nproc_per_node=1 src/train_classifier.py \
    --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set_our_format/ \
    --save_dir /mnt/raid/diamond_v3/${game}/classifier_512k_${model}/ \
    --train_n_examples 512000 \
    --batch_size 512 \
    --validation_batch_size 1024 \
    --num_classes ${num_classes} \
    --num_input_images 30 \
    --validation_subset_n 10000 \
    --validation_steps 250 \
    --save_steps 250 \
    --wandb_name ${game}_512k_${model} \
    --model ${model} \
    --batch_size 2048 \
    --lr 5e-4 \
    --training_data_buffer_size_gb 150 \
    --training_data_buffer_thread_count 24

# torchrun --nproc_per_node=1 src/train_classifier.py \
#     --dataset_path /mnt/raid/diamond_v3/${game}/fully_labeled_training_set_our_format/ \
#     --save_dir /mnt/raid/diamond_v3/${game}/classifier_1000/ \
#     --train_n_examples 1000 \
#     --batch_size 512 \
#     --validation_batch_size 1024 \
#     --num_classes ${num_classes} \
#     --num_input_images 30 \
#     --validation_subset_n 10000 \
#     --validation_steps 1000 \
#     --save_steps 1000 \
#     --wandb_name ${game}_1000