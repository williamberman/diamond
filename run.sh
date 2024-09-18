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
    --recording-dir ./${game}_recordings/ \
    --device cuda:${device}

python src/train_action_labeler.py \
    --epochs 3 \
    --checkpoint_dir ${game}_action_labelers \
    --data_dir ./${game}_recordings/ \
    --gpu ${device} 

python src/main.py \
    env.train.id=${game}NoFrameskip-v4 \
    collection.train.action_labeler_checkpoint="/workspace/${game}_action_labelers/action_labeler_final.pt" \
    common.device=cuda:${device} 