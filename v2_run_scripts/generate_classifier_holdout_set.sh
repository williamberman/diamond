#! /bin/bash

set -e
set -u

# game=$1
# gpu=$2
# has_negative_rewards=$3

game="Breakout"
gpu=0
has_negative_rewards=0

python src/play.py \
    --pretrained \
    --record \
    --recording-dir /mnt/raid/diamond_v2/${game}/classifier_holdout_set \
    --game ${game} \
    --eps dynamic \
    --eps-min 0.0 \
    --eps-max 0.75 \
    --eps-length 75 \
    --has-negative-rewards ${has_negative_rewards} \
    --store-final-obs \
    --headless-collect-until-min-things \
    --headless-collect-until-min-actions 10000 \
    --headless-collect-until-min-rewards 1000 \
    --device cuda:${gpu} \
    --default-env test
