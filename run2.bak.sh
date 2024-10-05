#! /bin/bash

set -e

# game=MsPacman
# gpu=0
# ./run2.sh MsPacman 0
# ./run2.sh Qbert 1
# ./run2.sh Alien 2
game=$1
gpu=$2

# python src/play.py \
#     --pretrained \
#     --record \
#     --recording-dir /mnt/raid/diamond/better4/${game}_recordings_100k \
#     --game ${game} \
#     --headless-collect-n-steps 100000 \
#     --eps 0.1 \
#     --store-final-obs \
#     --device cuda:${gpu}

# python src/train_action_labeler.py \
#     --epochs 1000 \
#     --checkpoint_dir /mnt/raid/diamond/better4/${game}_action_labelers_100k_1000 \
#     --data_dir /mnt/raid/diamond/better4/${game}_recordings_100k/ \
#     --train_size 0.01 \
#     --write_new_dataset_dir /mnt/raid/diamond/better4/${game}_recordings_100k_labeled_1000/ \
#     --eval_every_n_epochs 200 \
#     --lr 1e-4 \
#     --gpu ${gpu}

game="Qbert"
gpu=0

game="Breakout"
gpu=0

python src/train_action_labeler.py \
    --epochs 1000 \
    --data_dir /mnt/raid/diamond/better4/${game}_recordings_100k/ \
    --train_size 0.01 \
    --eval_every_n_epochs 1 \
    --lr 1e-5 \
    --gpu ${gpu}

# python src/main.py \
#     env.train.id=${game}NoFrameskip-v4 \
#     hydra.run.dir=/mnt/raid/diamond/better4/${game}_100k_labeled_1000_denoiser \
#     wandb.name=${game}_100k_labeled_1000_denoiser \
#     collection.path_to_static_dataset=/mnt/raid/diamond/better4/${game}_recordings_100k_labeled_1000/ \
#     common.device=cuda:${gpu} \
#     denoiser.train=True \
#     rew_end_model.train=True \
#     actor_critic.train=False

# python src/main.py \
#     env.train.id=${game}NoFrameskip-v4 \
#     hydra.run.dir=/mnt/raid/diamond/better4/${game}_100k_labeled_1000_actor_critic \
#     wandb.name=${game}_100k_labeled_1000_actor_critic \
#     collection.path_to_static_dataset=/mnt/raid/diamond/better4/${game}_recordings_100k_labeled_1000/ \
#     common.device=cuda:${gpu} \
#     denoiser.train=False \
#     rew_end_model.train=False \
#     actor_critic.train=True \
#     initialization.path_to_ckpt=/mnt/raid/diamond/better4/${game}_100k_labeled_1000_denoiser/checkpoints/agent_versions/agent_epoch_01000.pt