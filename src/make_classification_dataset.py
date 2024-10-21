# Take the index file, load and shuffle the dataset and write it out

import argparse
from data import Dataset as DiamondDataset
import torch
import tqdm
import os
import random
import numpy as np
import concurrent.futures

torch.set_grad_enabled(False)

def main():
    global args, dataset_indices

    parser = argparse.ArgumentParser(description="Make a dataset from indices")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument('--num_input_images', type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_obs = []
    all_cond = []
    all_target = []

    os.makedirs(args.output_path, exist_ok=True)

    assert len(os.listdir(args.output_path)) == 0

    pbar = tqdm.tqdm(desc="Loading dataset", total=get_total_episodes())

    for dataset_path in os.listdir(args.dataset_path):
        dataset_path = os.path.join(args.dataset_path, dataset_path)
        dataset = DiamondDataset(dataset_path)
        dataset.load_from_default_path()

        for episode_idx in range(dataset.num_episodes):
            episode = dataset.load_episode(episode_idx)

            for i in range(episode.obs.size(0) - args.num_input_images + 1):
                obs = episode.obs[i:i+args.num_input_images].flatten(0, 1)
                actions = episode.act[i:i+args.num_input_images]
                rewards = episode.rew[i:i+args.num_input_images]
                ends = episode.end[i:i+args.num_input_images]
            
                target_idx = args.num_input_images // 2
            
                actions = torch.cat([actions[:target_idx], actions[target_idx+1:]])
                cond = torch.cat([actions, rewards, ends], dim=0)
                target = episode.act[i+target_idx]

                all_obs.append(obs.numpy())
                all_cond.append(cond.numpy())
                all_target.append(target.numpy())

            pbar.update(1)

    # shuffle
    assert len(all_obs) == len(all_cond)

    pbar.close()

    perm = [x for x in range(len(all_obs))]
    random.shuffle(perm)
    
    chunk_size = 1000
    pbar = tqdm.tqdm(desc="Saving dataset", total=(len(perm)//(chunk_size+1)) + 1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        def save_item(ctr, perm_idx):
            end = min(perm_idx + chunk_size, len(perm))
            idxes = perm[perm_idx:end]

            obs = np.stack([all_obs[idx] for idx in idxes])
            cond = np.stack([all_cond[idx] for idx in idxes])
            target = np.stack([all_target[idx] for idx in idxes])

            np.save(os.path.join(args.output_path, f"{ctr}_obs.npy"), obs)
            np.save(os.path.join(args.output_path, f"{ctr}_cond.npy"), cond)
            np.save(os.path.join(args.output_path, f"{ctr}_target.npy"), target)

        futures = [executor.submit(save_item, ctr, perm_idx) for ctr, perm_idx in enumerate(range(0, len(perm), chunk_size))]

        for _ in concurrent.futures.as_completed(futures):
            pbar.update(1)

    pbar.close()

def get_total_episodes():
    total_episodes = 0

    for dataset_path in os.listdir(args.dataset_path):
        dataset_path = os.path.join(args.dataset_path, dataset_path)
        dataset = DiamondDataset(dataset_path)
        dataset.load_from_default_path()

        total_episodes += dataset.num_episodes

    return total_episodes


if __name__ == "__main__":
    main()