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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from models import ImprovedCNN
from data import Dataset
import glob
import concurrent.futures
import tqdm

def load_data_from_parquet(directory):
    all_data = []
    for file in glob.glob(os.path.join(directory, "*.parquet")):
        df = pd.read_parquet(file)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def load_data_from_dataset(directory):
    def dataset_to_df(dataset):
        all_data = []

        for episode_id in range(dataset.num_episodes):
            episode = dataset.load_episode(episode_id)

            for i in range(len(episode)-1):
                state = episode.obs[i]
                next_state = episode.obs[i+1]
                action = episode.act[i].item()

                all_data.append({
                    'state_img': state,
                    'next_state_img': next_state,
                    'action': action
                })

        df = pd.DataFrame(all_data)

        return df

    if len([x for x in os.listdir(directory) if x.endswith('.pt')]) > 0:
        dataset = Dataset(directory)
        dataset.load_from_default_path()
        df = dataset_to_df(dataset)
    else:
        datasets = []

        dirs = [os.path.join(directory, x) for x in os.listdir(directory)]

        pbar = tqdm.tqdm(total=len(dirs))

        def do(dir):
            dataset = Dataset(dir)
            dataset.load_from_default_path()
            df = dataset_to_df(dataset)
            datasets.append(df)
            pbar.update(1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(do, x) for x in dirs]
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
        next_state = self.df.iloc[idx]['next_state_img']
        action = self.df.iloc[idx]['action']

        if not isinstance(state, torch.Tensor):
            # the torch tensors loaded from the dataset are already correctly normalized etc..
            state = torch.tensor(np.array(state.resize((64, 64))))
            state = state.permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)
        
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.tensor(np.array(next_state.resize((64, 64))))
            next_state = next_state.permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)

        delta = next_state - state

        return delta, action

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    model.train()
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    steps_per_epoch = len(train_loader)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            if step % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, Train Loss: {loss.item():.4f} grad_norm: {grad_norm.item():.4f}')

        if (epoch+1) % 5 == 0 or epoch == num_epochs - 1:
            train_loss, train_accuracy = evaluate_model(model, train_loader, criterion, device)
            test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        else:
            train_loss, train_accuracy = 0, 0
            test_loss, test_accuracy = 0, 0

        print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}% '
              f'Avg Train Loss over epoch: {total_train_loss / steps_per_epoch:.4f} ')
    
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

    if len(glob.glob(os.path.join(args.data_dir, "**/*.parquet"))) > 0:
        df = load_data_from_parquet(args.data_dir)
        df = preprocess_data(df)
    elif len(glob.glob(os.path.join(args.data_dir, "**/*.pt"))) > 0:
        df = load_data_from_dataset(args.data_dir)
    else:
        raise ValueError(f"No parquet or dataset files found in {args.data_dir}")

    print(f"Data {len(df)} datapoints {len(df)//args.batch_size} batches per epoch loaded in {time.perf_counter() - t0:.2f} seconds")

    train_df, test_df = train_test_split(df, train_size=args.train_size, random_state=42)

    print(f"Train size: {len(train_df)} Test size: {len(test_df)}")

    train_dataset = StateActionDataset(train_df)
    test_dataset = StateActionDataset(test_df)
    train_loader = DataLoader(train_dataset, num_workers=args.batch_size, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    num_classes = df['action'].max() + 1
    model = ImprovedCNN(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, train_loader, test_loader, criterion, optimizer, args.epochs, device)

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
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    main(args)
