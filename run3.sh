#! /bin/bash

set -e

# ./run3.sh Qbert 0 # 60.76%, 63.96 ... 62.65, 94.99
# ./run3.sh MsPacman 1 # 65.71%, 69.24 ... 67.87, 87.46
# ./run3.sh Alien 2 # 52%, 52.93 ... 53.26, 90.54
# ./run3.sh Breakout 3 # 61.33%, 69.72 ... 69.49, 96.36

game=$1
gpu=$2

if [ -d "/mnt/raid/diamond/better4/${game}_recordings_100k_labeled_1000_2/" ]; then
    echo "New dataset already exists"
else
    python src/train_action_labeler.py \
    --epochs 600 \
    --data_dir /mnt/raid/diamond/better4/${game}_recordings_100k/ \
    --checkpoint_dir /mnt/raid/diamond/better4/${game}_action_labelers_100k_1000_4/ \
    --train_size 0.01 \
    --eval_every_n_epochs 100 \
    --lr 1e-6 \
    --gpu ${gpu} \
    --batch_size 256 \
    --game ${game} \
    --write_new_dataset_dir /mnt/raid/diamond/better4/${game}_recordings_100k_labeled_1000_2/
fi

python src/main.py \
    hydra.run.dir=/mnt/raid/diamond/better4/${game}_100k_labeled_1000_denoiser \
    env.train.id=${game}NoFrameskip-v4 \
    wandb.mode=online\
    wandb.name=${game}_100k_labeled_1000_denoiser \
    collection.path_to_static_dataset=/mnt/raid/diamond/better4/${game}_recordings_100k_labeled_1000_2/ \
    common.device=cuda:${gpu} \
    denoiser.train=False \
    rew_end_model.train=True \
    actor_critic.train=False