#! /bin/bash

set -e
set -u

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=23456 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 512000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_512000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_512000_resnet_sweep_1 &

pid0=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=23457 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 512000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_512000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_512000_resnet_sweep_2 &

pid1=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=23458 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 512000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_512000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_512000_resnet_sweep_3 &

pid2=$!

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=23459 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 512000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_512000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_512000_resnet_sweep_4 &

pid3=$!

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=23460 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 512000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_512000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_512000_resnet_sweep_5 &

pid4=$!

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=23461 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 256000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_256000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_256000_resnet_sweep_1 &

pid5=$!

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=23462 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 256000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_256000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_256000_resnet_sweep_2 &

pid6=$!

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=23463 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 256000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_256000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_256000_resnet_sweep_3 &

pid7=$!

wait $pid0 || exit 1
wait $pid1 || exit 1
wait $pid2 || exit 1
wait $pid3 || exit 1
wait $pid4 || exit 1
wait $pid5 || exit 1
wait $pid6 || exit 1
wait $pid7 || exit 1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=23456 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 256000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_256000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_256000_resnet_sweep_4 &

pid8=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=23457 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 256000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_256000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_256000_resnet_sweep_5 &

pid9=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=23458 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 128000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_128000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_128000_resnet_sweep_1 &

pid10=$!

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=23459 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 128000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_128000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_128000_resnet_sweep_2 &

pid11=$!

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=23460 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 128000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_128000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_128000_resnet_sweep_3 &

pid12=$!

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=23461 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 128000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_128000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_128000_resnet_sweep_4 &

pid13=$!

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=23462 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 128000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_128000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_128000_resnet_sweep_5 &

pid14=$!

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=23463 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 64000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_64000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_64000_resnet_sweep_1 &

pid15=$!

wait $pid8 || exit 1
wait $pid9 || exit 1
wait $pid10 || exit 1
wait $pid11 || exit 1
wait $pid12 || exit 1
wait $pid13 || exit 1
wait $pid14 || exit 1
wait $pid15 || exit 1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=23456 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 64000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_64000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_64000_resnet_sweep_2 &

pid16=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=23457 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 64000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_64000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_64000_resnet_sweep_3 &

pid17=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=23458 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 64000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_64000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_64000_resnet_sweep_4 &

pid18=$!

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=23459 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 64000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_64000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_64000_resnet_sweep_5 &

pid19=$!

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=23460 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 32000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_32000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_32000_resnet_sweep_1 &

pid20=$!

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=23461 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 32000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_32000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_32000_resnet_sweep_2 &

pid21=$!

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=23462 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 32000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_32000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_32000_resnet_sweep_3 &

pid22=$!

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=23463 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 32000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_32000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_32000_resnet_sweep_4 &

pid23=$!

wait $pid16 || exit 1
wait $pid17 || exit 1
wait $pid18 || exit 1
wait $pid19 || exit 1
wait $pid20 || exit 1
wait $pid21 || exit 1
wait $pid22 || exit 1
wait $pid23 || exit 1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=23456 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 32000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_32000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_32000_resnet_sweep_5 &

pid24=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=23457 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 16000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_16000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_16000_resnet_sweep_1 &

pid25=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=23458 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 16000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_16000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_16000_resnet_sweep_2 &

pid26=$!

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=23459 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 16000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_16000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_16000_resnet_sweep_3 &

pid27=$!

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=23460 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 16000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_16000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_16000_resnet_sweep_4 &

pid28=$!

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=23461 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 16000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_16000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_16000_resnet_sweep_5 &

pid29=$!

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=23462 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 8000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_8000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_8000_resnet_sweep_1 &

pid30=$!

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=23463 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 8000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_8000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_8000_resnet_sweep_2 &

pid31=$!

wait $pid24 || exit 1
wait $pid25 || exit 1
wait $pid26 || exit 1
wait $pid27 || exit 1
wait $pid28 || exit 1
wait $pid29 || exit 1
wait $pid30 || exit 1
wait $pid31 || exit 1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=23456 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 8000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_8000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_8000_resnet_sweep_3 &

pid32=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=23457 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 8000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_8000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_8000_resnet_sweep_4 &

pid33=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=23458 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 8000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_8000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_8000_resnet_sweep_5 &

pid34=$!

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=23459 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 4000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_4000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_4000_resnet_sweep_1 &

pid35=$!

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=23460 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 4000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_4000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_4000_resnet_sweep_2 &

pid36=$!

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=23461 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 4000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_4000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_4000_resnet_sweep_3 &

pid37=$!

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=23462 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 4000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_4000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_4000_resnet_sweep_4 &

pid38=$!

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=23463 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 4000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_4000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_4000_resnet_sweep_5 &

pid39=$!

wait $pid32 || exit 1
wait $pid33 || exit 1
wait $pid34 || exit 1
wait $pid35 || exit 1
wait $pid36 || exit 1
wait $pid37 || exit 1
wait $pid38 || exit 1
wait $pid39 || exit 1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=23456 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 2000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_2000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_2000_resnet_sweep_1 &

pid40=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=23457 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 2000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_2000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_2000_resnet_sweep_2 &

pid41=$!

CUDA_VISIBLE_DEVICES=2 torchrun --nproc_per_node=1 --master_port=23458 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 2000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_2000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_2000_resnet_sweep_3 &

pid42=$!

CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=23459 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 2000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_2000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_2000_resnet_sweep_4 &

pid43=$!

CUDA_VISIBLE_DEVICES=4 torchrun --nproc_per_node=1 --master_port=23460 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 2000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_2000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_2000_resnet_sweep_5 &

pid44=$!

CUDA_VISIBLE_DEVICES=5 torchrun --nproc_per_node=1 --master_port=23461 src/train_classifier.py \
    --model_type resnet_sweep_1 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 1000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_1000_resnet_sweep_1/ \
    --save_epochs 1 \
    --wandb_name Breakout_1000_resnet_sweep_1 &

pid45=$!

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --master_port=23462 src/train_classifier.py \
    --model_type resnet_sweep_2 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 1000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_1000_resnet_sweep_2/ \
    --save_epochs 1 \
    --wandb_name Breakout_1000_resnet_sweep_2 &

pid46=$!

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port=23463 src/train_classifier.py \
    --model_type resnet_sweep_3 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 1000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_1000_resnet_sweep_3/ \
    --save_epochs 1 \
    --wandb_name Breakout_1000_resnet_sweep_3 &

pid47=$!

wait $pid40 || exit 1
wait $pid41 || exit 1
wait $pid42 || exit 1
wait $pid43 || exit 1
wait $pid44 || exit 1
wait $pid45 || exit 1
wait $pid46 || exit 1
wait $pid47 || exit 1

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=23456 src/train_classifier.py \
    --model_type resnet_sweep_4 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 1000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_1000_resnet_sweep_4/ \
    --save_epochs 1 \
    --wandb_name Breakout_1000_resnet_sweep_4 &

pid48=$!

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=23457 src/train_classifier.py \
    --model_type resnet_sweep_5 \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr 5e-4 \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/Breakout/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples 1000 \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/Breakout/classifier_1000_resnet_sweep_5/ \
    --save_epochs 1 \
    --wandb_name Breakout_1000_resnet_sweep_5 &

pid49=$!

wait $pid48 || exit 1
wait $pid49 || exit 1