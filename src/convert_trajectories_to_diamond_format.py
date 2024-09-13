import os
import pandas as pd
from PIL import Image
import io
import numpy as np
from data import Episode, Dataset
import torch
import cv2
from tqdm import tqdm
import concurrent.futures
from threading import Lock

def convert_parquet_to_episode(parquet_file):
    df = pd.read_parquet(parquet_file)
    obs = []
    act = []
    rew = []
    end = []
    trunc = []
    info = {}

    for _, row in df.iterrows():
        # Convert state and next_state from bytes to numpy arrays
        ob = np.array(Image.open(io.BytesIO(row['state'])).convert('RGB'))
        ob = cv2.resize(ob, (64, 64), interpolation=cv2.INTER_AREA)
        obs.append(torch.tensor(ob).permute(2, 0, 1).contiguous())
        act.append(torch.tensor(row['action']))
        rew.append(torch.tensor(row['reward_this_period']))
        end.append(torch.tensor(0))
        trunc.append(torch.tensor(0))

    obs = torch.stack(obs)
    act = torch.stack(act)
    rew = torch.stack(rew)
    end = torch.stack(end)
    trunc = torch.stack(trunc)

    print(f"obs: {obs.shape}, act: {act.shape}, rew: {rew.shape}, end: {end.shape}, trunc: {trunc.shape}")

    # Create an Episode object
    episode = Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info=info)
    return episode

def process_trajectory_samples():
    trajectory_dir = '/mnt/raid/orca_rl/trajectory_samples'

    os.system("rm -rf /mnt/raid/orca_rl/trajectory_samples_diamond_format/")

    train_parquets = [x for x in os.listdir(trajectory_dir) if x.endswith('.parquet')]
    test_parquets = [train_parquets.pop() for _ in range(10)]


    dataset = Dataset("/mnt/raid/orca_rl/trajectory_samples_diamond_format/train", name="train_dataset", save_on_disk=True, use_manager=False)

    lock = Lock()
    pbar = tqdm(total=len(train_parquets))

    def process_parquet(dataset, parquets, i):
        filename = parquets[i]
        parquet_path = os.path.join(trajectory_dir, filename)
        episode = convert_parquet_to_episode(parquet_path)

        try:
            lock.acquire()
            dataset.add_episode(episode)
            pbar.update(1)
        finally:
            lock.release()

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(lambda i: process_parquet(dataset,train_parquets, i), i) for i in range(len(train_parquets))]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    dataset.is_static = True
    dataset.save_to_default_path()

    test_dataset = Dataset("/mnt/raid/orca_rl/trajectory_samples_diamond_format/test", name="test_dataset", save_on_disk=True, use_manager=False)

    pbar = tqdm(total=len(test_parquets))

    with concurrent.futures.ThreadPoolExecutor(max_workers=128) as executor:
        futures = [executor.submit(lambda i: process_parquet(test_dataset,test_parquets, i), i) for i in range(len(test_parquets))]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    test_dataset.is_static = True
    test_dataset.save_to_default_path()

if __name__ == "__main__":
    process_trajectory_samples()
