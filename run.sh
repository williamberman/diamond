#! /bin/bash

set -e

game=$1
device=$2

python src/play.py \
    --pretrained \
    --record \
    --default-env test \
    --headless-collect-n 400 \
    --game $game \
    --recording-dir /mnt/raid/diamond/better/${game}_recordings/ \
    --device cuda:${device}

python src/train_action_labeler.py \
    --epochs 3 \
    --checkpoint_dir /mnt/raid/diamond/better/${game}_action_labelers \
    --data_dir /mnt/raid/diamond/better/${game}_recordings/ \
    --gpu ${device} 

python src/main.py \
    env.train.id=${game}NoFrameskip-v4 \
    hydra.run.dir=/mnt/raid/diamond/better/${game} \
    collection.train.action_labeler_checkpoint="/mnt/raid/diamond/better/${game}_action_labelers/action_labeler_final.pt" \
    wandb.name=${game} \
    common.device=cuda:${device} 