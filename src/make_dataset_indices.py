import argparse
import os
import random
from data import Dataset as DiamondDataset
import json
import tqdm
# This is where we generate the dataset indices that will define a split of approximately <x> frames
# It should be consistent from call to call with the same random seed. The indices are the the path
# to the main dataset and the index of the episodes in each dataset.
def main():
    global args

    parser = argparse.ArgumentParser(description="Generate dataset indices")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--frame_count", type=int, required=True, help="Approximate number of frames for the dataset split")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    random.seed(args.seed)

    # each top level directory is its own dataset

    datasets = [os.path.join(args.dataset_path, d) for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]

    print(f"Found {len(datasets)} datasets")

    for dataset in datasets:
        print(dataset)

    frame_count = 0
    n_episodes = 0
    frame_indices = {d: [] for d in datasets}

    pbar = tqdm.tqdm(total=args.frame_count)

    while True:
        # select a random dataset
        dataset_idx = random.randint(0, len(datasets) - 1)

        # load the dataset
        dataset = DiamondDataset(datasets[dataset_idx])
        assert dataset.load_from_default_path()
        dataset._cache_in_ram = False

        # select a random episode from the dataset
        random_episode_idx = random.randint(0, dataset.num_episodes - 1)

        # count the frames in the episode
        episode = dataset.load_episode(random_episode_idx)
        episode_len = len(episode)

        # add the episode to the list of frame indices
        frame_indices[datasets[dataset_idx]].append(dict(episode_idx=random_episode_idx, episode_len=episode_len))

        # update the total frame count
        frame_count += episode_len
        n_episodes += 1

        # update the progress bar
        pbar.update(episode_len)
        pbar.set_postfix(n_episodes=n_episodes)

        # check if we've reached the target frame count
        if frame_count >= args.frame_count:
            break

    results = dict(
        frame_indices=frame_indices,
        frame_count=frame_count,
        n_episodes=n_episodes,
    )

    # save the results
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()