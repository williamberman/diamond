#! /bin/bash

set -e
set -u

# game=$1
# gpu=$2
# has_negative_rewards=$3

game="Breakout"
gpu=0
has_negative_rewards=0

root_dir=/mnt/raid/diamond_v3

python src/play.py \
    --pretrained \
    --record \
    --recording-dir ${root_dir}/${game}/fully_labeled_training_set \
    --game ${game} \
    --eps dynamic-levels \
    --eps-length 75\
    --has-negative-rewards ${has_negative_rewards} \
    --store-final-obs \
    --headless-collect-n-steps 1000000 \
    --device cuda:${gpu} \
    --default-env test