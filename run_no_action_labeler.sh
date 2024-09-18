#! /bin/bash

set -e

game=$1
device=$2

python src/main.py \
    env.train.id=${game}NoFrameskip-v4 \
    hydra.run.dir=/mnt/raid/diamond/better/${game}_no_action_labeler \
    collection.train.action_labeler=null \
    wandb.name=${game}_no_action_labeler \
    common.device=cuda:${device} 