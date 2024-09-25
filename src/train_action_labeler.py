import os
import time
import io
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from models import AnotherCNN
from data import Dataset, Episode
import glob
import concurrent.futures
import tqdm
from collections import defaultdict
from PIL import ImageDraw
import gymnasium as gym

context_n_frames = 10
assert context_n_frames == 10

def load_data_from_parquet(directory):
    all_data = []
    for file in glob.glob(os.path.join(directory, "*.parquet")):
        df = pd.read_parquet(file)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_data_from_dataset(directory):
    def dataset_to_df(dir):
        dataset = Dataset(dir)
        dataset.load_from_default_path()

        all_data = []

        for episode_id in range(dataset.num_episodes):
            episode = dataset.load_episode(episode_id)

            for predicting_for_episode_idx in range(len(episode)):
                state_imgs = []

                for i in [4, 3, 2, 1]:
                    if predicting_for_episode_idx - i >= 0:
                        state_imgs.append(episode.obs[predicting_for_episode_idx - i])
                    else:
                        state_imgs.append(torch.zeros_like(episode.obs[0]))

                image_predicting_for = episode.obs[predicting_for_episode_idx]
                state_imgs.append(episode.obs[predicting_for_episode_idx])
                action = episode.act[predicting_for_episode_idx].item()

                for i in [1,2,3,4,5]:
                    if predicting_for_episode_idx + i < len(episode):
                        state_imgs.append(episode.obs[predicting_for_episode_idx + i])
                    else:
                        state_imgs.append(torch.zeros_like(episode.obs[0]))

                assert len(state_imgs) == context_n_frames

                all_data.append({
                    'state_img': state_imgs,
                    'image_predicting_for': image_predicting_for,
                    'action': action,
                    'episode_id': episode_id,
                    'reward': episode.rew[predicting_for_episode_idx],
                    'end': episode.end[predicting_for_episode_idx],
                    'trunc': episode.trunc[predicting_for_episode_idx],
                    'dir': dir,
                    'step_id': predicting_for_episode_idx,
                })

        df = pd.DataFrame(all_data)

        return df

    if len([x for x in os.listdir(directory) if x.endswith('.pt')]) > 0:
        df = dataset_to_df(directory)
    else:
        # sort the directories and add them in the same order so that 
        # splitting the dataframe later will result in the same
        # train/test split order across different runs. Also requires
        # the same random state for the train/test split

        dirs = [os.path.join(directory, x) for x in os.listdir(directory)]
        dirs = sorted(dirs)

        datasets = [None] * len(dirs)

        pbar = tqdm.tqdm(total=len(dirs))

        def do(ctr, dir):
            df = dataset_to_df(dir)
            datasets[ctr] = df
            pbar.update(1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(do, ctr, dir) for ctr, dir in enumerate(dirs)]
            for future in concurrent.futures.as_completed(futures):
                future.result()

        df = pd.concat(datasets, ignore_index=True)

    return df

def preprocess_data(df):
    def bytes_to_image(byte_data):
        return Image.open(io.BytesIO(byte_data))
    
    df['state_img'] = df['state'].apply(bytes_to_image)
    df['next_state_img'] = df['next_state'].apply(bytes_to_image)
    return df

class StateActionDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        state = self.df.iloc[idx]['state_img']
        action = self.df.iloc[idx]['action']
        state = torch.stack(state)
        t, c, h, w = state.shape
        state = state.reshape((t*c, h, w))
        return state, action

@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device, id, take=None):
    if args.dbg:
        action_strs = gym.make(args.game).get_action_meanings()

    model.eval() # TODO - wtf
    total_loss = 0
    correct = 0
    total = 0
    ctr = 0

    for inputs, labels in tqdm.tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        ctr += 1

        if args.dbg:
            wrong_predictions_dir = os.path.join(args.checkpoint_dir, id, f"wrong_predictions")
            right_predictions_dir = os.path.join(args.checkpoint_dir, id, f"right_predictions")

            os.makedirs(wrong_predictions_dir, exist_ok=True)
            os.makedirs(right_predictions_dir, exist_ok=True)

            for i in range(len(inputs)):
                input = inputs[i]
                prediction = predicted[i]
                label = labels[i]

                # turn inputs into images
                input = input.mul(0.5).add(0.5).mul(255).clamp(0, 255).byte().reshape(context_n_frames, 3, 64, 64).permute(0, 2, 3, 1).cpu().numpy()
                input = [Image.fromarray(x).resize((256, 256)) for x in input]

                # paste all the images into a single image
                paste_into = Image.new('RGB', (256*(context_n_frames+1), 256))
                for j, img in enumerate(input):
                    if j <= 5:
                        paste_into.paste(img, (256*j, 0))
                    else:
                        paste_into.paste(img, (256*(j+1), 0))

                prediction_action_text = f"predicted: {action_strs[prediction.item()]}"
                actual_action_text = f"actual: {action_strs[label.item()]}"

                paste_into_ = paste_into.copy()

                draw = ImageDraw.Draw(paste_into_)
                draw.text((256*(6)+20, 64), prediction_action_text, fill=(255, 255, 255))
                draw.text((256*(6)+20, 64+128), actual_action_text, fill=(255, 255, 255))

                # save the input images
                if prediction == label:
                    paste_into_.save(os.path.join(right_predictions_dir, f"{ctr}_{i}.png"))
                else:
                    paste_into_.save(os.path.join(wrong_predictions_dir, f"{ctr}_{i}.png"))

        if take is not None and ctr >= take:
            break
    
    avg_loss = total_loss / ctr
    accuracy = 100 * correct / total
    model.train()
    print(f"Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    steps_per_epoch = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        train_loss_ctr = 0
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_loss_ctr += 1
            if step % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, Train Loss: {loss.item():.4f} grad_norm: {grad_norm.item():.4f}')

        if (epoch+1) % args.eval_every_n_epochs == 0 or epoch == num_epochs - 1:
            train_loss, train_accuracy = evaluate_model(model, train_loader, criterion, device, id=f"{epoch}_train")
            test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device, id=f"{epoch}_test")
        else:
            train_loss, train_accuracy = 0, 0
            test_loss, test_accuracy = 0, 0

        print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}% '
              f'Avg Train Loss over epoch: {total_train_loss / train_loss_ctr:.4f} ')
    
        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"action_labeler_{epoch+1}.pt"))

    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"action_labeler_final.pt"))

def main(args):
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    t0 = time.perf_counter()
    print(f"Loading data from {args.data_dir}")

    if len(glob.glob(os.path.join(args.data_dir, "**/*.parquet"))) > 0 or len(glob.glob(os.path.join(args.data_dir, "*.parquet"))) > 0:
        df = load_data_from_parquet(args.data_dir)
        df = preprocess_data(df)
    elif len(glob.glob(os.path.join(args.data_dir, "**/*.pt"))) > 0 or len(glob.glob(os.path.join(args.data_dir, "*.pt"))) > 0:
        df = load_data_from_dataset(args.data_dir)
    else:
        raise ValueError(f"No parquet or dataset files found in {args.data_dir}")

    print(f"Data {len(df)} datapoints {len(df)//args.batch_size} batches per epoch loaded in {time.perf_counter() - t0:.2f} seconds")

    train_df, test_df = train_test_split(df, train_size=args.train_size, random_state=42)
    train_df['split'] = ['train']*len(train_df)
    test_df['split'] = ['test']*len(test_df)

    print(f"Train size: {len(train_df)} Test size: {len(test_df)}")

    train_dataset = StateActionDataset(train_df)
    test_dataset = StateActionDataset(test_df)
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    num_classes = df['action'].max() + 1
    # model = SimpleCNN(context_n_frames=context_n_frames, num_classes=num_classes).to(device).train()
    model = AnotherCNN(context_n_frames=context_n_frames, num_classes=num_classes).to(device).train()
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"action_labeler_final.pt")))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, train_loader, test_loader, criterion, optimizer, args.epochs, device)

    # evaluate_model(model, train_loader, criterion, device, id="final_train")
    # evaluate_model(model, test_loader, criterion, device, id="final_test")

    if args.write_new_dataset_dir is not None:
        write_new_dataset(train_df, test_df, model)

@torch.no_grad()
def write_new_dataset(train_df, test_df, model):
    model.eval()

    ds = Dataset(args.write_new_dataset_dir)

    df = pd.concat([train_df, test_df], ignore_index=True)

    pbar = tqdm.tqdm(total=len(df))

    episodes = defaultdict(list)
    episode_data = {}

    num_right = 0

    for dir in df.dir.unique():
        dir_df = df[df.dir == dir]

        for episode_id in dir_df.episode_id.unique():
            episode_data[f"{dir}_{episode_id}"] = [dir, episode_id]

            episode_df = dir_df[dir_df.episode_id == episode_id]

            step_ids = sorted(episode_df.step_id.tolist())
            max_step_id = step_ids[-1]
            assert step_ids == list(range(max_step_id+1))

            for step_id in step_ids:
                row = episode_df[episode_df.step_id == step_id].iloc[0]

                obs = row['image_predicting_for']
                act = row['action']
                episode_id = row['episode_id']
                rew = row['reward']
                end = row['end']
                trunc = row['trunc']

                if row['split'] == 'test':
                    state_img = torch.stack(row['state_img'])
                    t, c, h, w = state_img.shape
                    model_input = state_img.reshape(t*c, h, w).unsqueeze(0).to(args.gpu)
                    act = model(model_input).argmax().item()
                    if act == row['action']:
                        num_right += 1

                episodes[f"{dir}_{episode_id}"].append((obs, act, rew, end, trunc))

                pbar.update(1)

    pbar.close()

    pbar = tqdm.tqdm(total=len(episodes))
    datasets = {}

    for episode_key, episode in episodes.items():
        dir, episode_id = episode_data[episode_key]

        if dir not in datasets:
            datasets[dir] = Dataset(dir)
            datasets[dir].load_from_default_path()

        dataset = datasets[dir]
        info = dataset.load_episode(episode_id).info

        obs = []
        act = []
        rew = []
        end = []
        trunc = []

        for step in episode:
            obs.append(step[0])
            act.append(step[1])
            rew.append(step[2])
            end.append(step[3])
            trunc.append(step[4])

        obs = torch.stack(obs)
        act = torch.tensor(act, dtype=torch.long)
        rew = torch.stack(rew)
        end = torch.stack(end)
        trunc = torch.stack(trunc)

        assert obs.ndim == 4
        assert act.ndim == 1
        assert rew.ndim == 1
        assert end.ndim == 1
        assert trunc.ndim == 1

        assert obs.dtype == torch.float32
        assert act.dtype == torch.int64
        assert rew.dtype == torch.float32
        assert end.dtype == torch.uint8
        assert trunc.dtype == torch.uint8

        assert obs.shape[0] == act.shape[0] 
        assert obs.shape[0] == rew.shape[0] 
        assert obs.shape[0] == end.shape[0] 
        assert obs.shape[0] == trunc.shape[0]

        episode = Episode(obs, act, rew, end, trunc, info)

        ds.add_episode(episode)

        pbar.update(1)

    pbar.close()

    print(f"Num right: {num_right}/{len(test_df)} {num_right/len(test_df):.2f}")

    print(f"len(df): {len(df)} len(ds): {len(ds)}")

    assert len(df) == len(ds)

    ds.is_static = True
    ds.save_to_default_path()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classifier for state-action prediction")
    parser.add_argument("--data_dir", type=str, default="/mnt/raid/orca_rl/trajectory_samples", help="Directory containing parquet files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=160, help="Height to resize images to (width will be adjusted to maintain aspect ratio)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gpu", type=int, default=7, help="GPU ID to use (default: None, use CPU)")
    parser.add_argument("--checkpoint_dir", type=str, default="action_labelers", help="Directory to save checkpoints")
    parser.add_argument("--train_size", type=float, default=0.95, help="Proportion of data to use for training")
    parser.add_argument("--write_new_dataset_dir", type=str, default=None, help="If set, write a new dataset to this directory")
    parser.add_argument("--eval_every_n_epochs", type=int, default=5, help="Evaluate the model every n epochs")
    parser.add_argument("--game", type=str, default=None, required=False, help="Game to evaluate on")
    parser.add_argument("--dbg", action='store_true', help="Debug mode")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.write_new_dataset_dir is not None:
        if os.path.exists(args.write_new_dataset_dir):
            print(f"WARNING: {args.write_new_dataset_dir} already exists. IDK what diamond does with that. You probably just want to delete it first to be sure you cleanly overwrite it.")

    main(args)
