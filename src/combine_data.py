from data import Dataset
import os
import tqdm

def main():
    dir = "/mnt/raid/diamond/action_autoencoder/dataset/Breakout_recordings_100k/"
    new_dataset = Dataset("/mnt/raid/diamond/action_autoencoder/dataset/Breakout_recordings_100k_correct/")
    # new_dataset.load_from_default_path()
    # import ipdb; ipdb.set_trace()

    for x in os.listdir(dir):
        print(x)
        old_dataset = Dataset(os.path.join(dir, x))
        old_dataset.load_from_default_path()

        for i in tqdm.tqdm(range(old_dataset.num_episodes)):
            new_dataset.add_episode(old_dataset.load_episode(i))

    new_dataset.is_static = True
    new_dataset.save_to_default_path()

if __name__ == "__main__":
    main()

