import os
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

def load_data(directory):
    all_data = []
    for file in glob.glob(os.path.join(directory, "*.parquet")):
        df = pd.read_parquet(file)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

def preprocess_data(df):
    def bytes_to_image(byte_data):
        return Image.open(io.BytesIO(byte_data))
    
    df['state_img'] = df['state'].apply(bytes_to_image)
    df['next_state_img'] = df['next_state'].apply(bytes_to_image)
    return df

class StateActionDataset(Dataset):
    def __init__(self, df, transform=None, use_delta=False):
        self.df = df
        self.transform = transform
        self.use_delta = use_delta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        state = self.df.iloc[idx]['state_img']
        next_state = self.df.iloc[idx]['next_state_img']
        action = self.df.iloc[idx]['action']

        if self.transform:
            state = self.transform(state)
            next_state = self.transform(next_state)

        if self.use_delta:
            delta = next_state - state
            return delta, action
        else:
            return torch.cat((state, next_state), dim=0), action

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(ImprovedCNN, self).__init__()
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        self.layer1 = self._make_layer(32, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

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
    return avg_loss, accuracy

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device):
    steps_per_epoch = len(train_loader)
    eval_interval = steps_per_epoch // 10  # Evaluate every 1/10 of an epoch

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for step, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            if step % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, Train Loss: {loss.item():.4f}')

            if step % eval_interval == 0 or step == steps_per_epoch:
                test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
                print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, '
                      f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
                model.train()  # Switch back to training mode

        avg_train_loss = total_train_loss / steps_per_epoch
        print(f'Epoch {epoch+1}/{num_epochs} completed, Average Train Loss: {avg_train_loss:.4f}')
        
        scheduler.step()  # Step the learning rate scheduler

def main(args):
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_data(args.data_dir)
    df = preprocess_data(df)

    original_img_size = df['state_img'].iloc[0].size
    print(f"Original image size: {original_img_size}")

    # Keep aspect ratio
    aspect_ratio = original_img_size[0] / original_img_size[1]
    new_height = args.image_size
    new_width = int(new_height * aspect_ratio)

    # Use standard normalization values
    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_df, test_df = train_test_split(df, test_size=0.05, random_state=42)

    train_dataset = StateActionDataset(train_df, transform=transform, use_delta=args.use_delta)
    test_dataset = StateActionDataset(test_df, transform=transform, use_delta=args.use_delta)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    num_classes = df['action'].nunique()
    input_channels = 3 if args.use_delta else 6
    model = ImprovedCNN(num_classes, input_channels).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, args.epochs, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an image classifier for state-action prediction")
    parser.add_argument("--data_dir", type=str, default="/mnt/raid/orca_rl/trajectory_samples", help="Directory containing parquet files")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--image_size", type=int, default=160, help="Height to resize images to (width will be adjusted to maintain aspect ratio)")
    parser.add_argument("--use_delta", action="store_true", help="Include delta between state and next_state as input")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gpu", type=int, default=2, help="GPU ID to use (default: None, use CPU)")
    args = parser.parse_args()

    main(args)