#! /bin/bash

set -e

# game=MsPacman
# gpu=0
# ./run2.sh MsPacman 0
# ./run2.sh Qbert 1
# ./run2.sh Alien 2
# ./run2.sh Breakout 3
# ./run2.sh Pong 4
# ./run2.sh Asterix 5
game=$1
gpu=$2

# python src/play.py \
#     --pretrained \
#     --record \
#     --recording-dir /mnt/raid/diamond/tiny/${game}_recordings_100k \
#     --game ${game} \
#     --headless-collect-n-steps 100000 \
#     --eps 0.1 \
#     --store-final-obs \
#     --device cuda:${gpu}

python src/train_action_labeler.py \
    --epochs 600 \
    --data_dir /mnt/raid/diamond/tiny/${game}_recordings_100k/ \
    --checkpoint_dir /mnt/raid/diamond/tiny/${game}_action_labelers_100k_1000 \
    --train_size 0.01 \
    --eval_every_n_epochs 100 \
    --lr 1e-6 \
    --gpu ${gpu} \
    --batch_size 64 \
    --game ${game} \
    --write_new_dataset_dir /mnt/raid/diamond/tiny/${game}_recordings_100k_labeled_1000/

exit 0

python src/main.py \
    hydra.run.dir=/mnt/raid/diamond/tiny/${game}_100k_labeled_1000_denoiser \
    env.train.id=${game}NoFrameskip-v4 \
    wandb.mode=online \
    wandb.name=${game}_100k_labeled_1000_denoiser \
    collection.path_to_static_dataset=/mnt/raid/diamond/tiny/${game}_recordings_100k_labeled_1000/ \
    common.device=cuda:${gpu} \
    denoiser.train=True \
    rew_end_model.train=True \
    actor_critic.train=False

python src/main.py \
    hydra.run.dir=/mnt/raid/diamond/tiny/${game}_100k_labeled_1000_actor_critic \
    env.train.id=${game}NoFrameskip-v4 \
    wandb.mode=online \
    wandb.name=${game}_100k_labeled_1000_actor_critic \
    collection.path_to_static_dataset=/mnt/raid/diamond/tiny/${game}_recordings_100k_labeled_1000/ \
    common.device=cuda:${gpu} \
    denoiser.train=False \
    rew_end_model.train=False \
    actor_critic.train=True \
    initialization.path_to_ckpt=/mnt/raid/diamond/tiny/${game}_100k_labeled_1000_denoiser/checkpoints/agent_versions/agent_epoch_01000.pt