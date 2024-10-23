from argparse import ArgumentParser
import os

def main():
    args = ArgumentParser()
    args.add_argument("--game", type=str, required=True)
    args.add_argument("--n_concurrent_processes", type=int, default=8)
    args = args.parse_args()


    script = """#! /bin/bash

set -e
set -u
"""

    ctr = 0
    pids = []

    # dataset_train_split_n_examples = [512000, 256000, 128000, 64000, 32000, 16000, 8000, 4000, 2000, 1000]
    dataset_train_split_n_examples = [128000, 64000, 32000, 16000, 8000, 4000, 2000, 1000]
    model_types = ["resnet_sweep_1", "resnet_sweep_2", "resnet_sweep_3", "resnet_sweep_4", "resnet_sweep_5"]
    # lr = 5e-4
    lr = 5e-5

    for dataset_train_split_n_example in dataset_train_split_n_examples:
        for model_type in model_types:
            cmd = rf"""
CUDA_VISIBLE_DEVICES={ctr%args.n_concurrent_processes} torchrun --nproc_per_node=1 --master_port={23456+(ctr%args.n_concurrent_processes)} src/train_classifier.py \
    --model_type {model_type} \
    --model_num_classes 4 \
    --model_num_input_images 30 \
    --model_lr {lr} \
    --model_batch_size 2048 \
    --dataset_path /mnt/raid/diamond_v3/{args.game}/fully_labeled_training_set_our_format/ \
    --dataset_train_split_n_examples {dataset_train_split_n_example} \
    --dataset_buffer_size_gb 220 \
    --dataset_fill_buffer_thread_count 20 \
    --validation_n_examples 10000 \
    --validation_batch_size 4096 \
    --validation_epochs 1 \
    --save_dir /mnt/raid/diamond_v3/{args.game}/classifier_{dataset_train_split_n_example}_{model_type}_lr_{lr}/ \
    --save_epochs 1 \
    --wandb_name {args.game}_{dataset_train_split_n_example}_{model_type}_lr_{lr} &

pid{ctr}=$!
"""

            pids.append(ctr)

            script += cmd

            ctr += 1

            if ctr % args.n_concurrent_processes == 0:
                script += "\n"
                for pid in pids:
                    script += f"wait $pid{pid} || exit 1\n"
                pids = []

    if len(pids) > 0:
        script += "\n"
        for pid in pids:
            script += f"wait $pid{pid} || exit 1\n"

    with open(f"v3_run_scripts/train_classifier_sweep_{args.game}_lr_{lr}.sh", "w") as f:
        f.write(script)

    os.system(f"chmod +x v3_run_scripts/train_classifier_sweep_{args.game}_lr_{lr}.sh")

if __name__ == "__main__":
    main()
