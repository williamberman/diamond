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
import torch.nn.functional as F

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
    def __init__(self, df, only_non_zero_rewards=False):
        if only_non_zero_rewards:
            rewards = torch.stack(df.reward.tolist())
            df = df[(rewards!=0).flatten().tolist()]

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        state = self.df.iloc[idx]['state_img']
        action = self.df.iloc[idx]['action']

        reward = self.df.iloc[idx]['reward']
        reward = reward.sign().to(torch.long) + 1
        assert reward.item() in [0, 1, 2]

        state = torch.stack(state)
        t, c, h, w = state.shape
        state = state.reshape((t*c, h, w))

        dir_id = self.df.iloc[idx]['dir'].split('/')[-1]
        unique_id = f"{dir_id}_{self.df.iloc[idx]['episode_id']}_{self.df.iloc[idx]['step_id']}"

        return dict(
            state=state,
            action=action,
            reward=reward,
            unique_id=unique_id,
        )

@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device, id, take=None):
    if args.dbg:
        action_strs = gym.make(args.game).get_action_meanings()

    model.eval()

    total_loss = 0
    total_loss_actions = 0
    total_loss_rewards = 0

    correct = 0
    correct_actions = 0
    correct_rewards = 0

    total = 0
    ctr = 0

    true_pos_reward_expected_reward = []
    true_neutral_reward_expected_reward = []
    true_neg_reward_expected_reward = []

    for it in tqdm.tqdm(data_loader):
        inputs = it['state']
        action_labels = it['action']
        reward_labels = it['reward']

        unique_ids = it['unique_id']

        inputs, action_labels, reward_labels = inputs.to(device), action_labels.to(device), reward_labels.to(device)
        outputs_actions, outputs_rewards = model(inputs)

        outputs_rewards_probs = F.softmax(outputs_rewards, dim=1)
        neg_reward_probs = outputs_rewards_probs[:, 0]
        pos_reward_probs = outputs_rewards_probs[:, 2]

        expected_reward = pos_reward_probs - neg_reward_probs
        true_pos_reward_expected_reward.extend(expected_reward[reward_labels == 2].flatten().tolist())
        true_neutral_reward_expected_reward.extend(expected_reward[reward_labels == 1].flatten().tolist())
        true_neg_reward_expected_reward.extend(expected_reward[reward_labels == 0].flatten().tolist())

        loss_actions = criterion(outputs_actions, action_labels)

        loss_rewards = criterion(outputs_rewards, reward_labels)

        loss = loss_actions + loss_rewards

        total_loss += loss.item()
        total_loss_actions += loss_actions.item()
        total_loss_rewards += loss_rewards.item()

        _, predicted_actions = torch.max(outputs_actions.data, 1)
        _, predicted_rewards = torch.max(outputs_rewards.data, 1)

        total += action_labels.size(0)

        correct += ((predicted_actions == action_labels) & (predicted_rewards == reward_labels)).sum().item()

        correct_actions += (predicted_actions == action_labels).sum().item()

        correct_rewards += (predicted_rewards == reward_labels).sum().item()

        ctr += 1

        if args.dbg:
            wrong_predictions_dir = os.path.join(args.checkpoint_dir, id, f"wrong_predictions")
            right_predictions_dir = os.path.join(args.checkpoint_dir, id, f"right_predictions")

            os.makedirs(wrong_predictions_dir, exist_ok=True)
            os.makedirs(right_predictions_dir, exist_ok=True)

            for i in range(len(inputs)):
                input = inputs[i]

                prediction_action = predicted_actions[i]
                prediction_reward = predicted_rewards[i]

                label_action = action_labels[i]
                label_reward = reward_labels[i]

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

                prediction_action_text = f"predicted action: {action_strs[prediction_action.item()]}"
                actual_action_text = f"actual action: {action_strs[label_action.item()]}"

                reward_strs = ['negative reward', 'no reward', 'positive reward']
                prediction_reward_text = f"predicted reward: {reward_strs[prediction_reward.item()]}"
                actual_reward_text = f"actual reward: {reward_strs[label_reward.item()]}"

                paste_into_ = paste_into.copy()

                draw = ImageDraw.Draw(paste_into_)
                draw.text((256*(6)+20, 10), prediction_action_text, fill=(255, 255, 255))
                draw.text((256*(6)+20, 10+64), actual_action_text, fill=(255, 255, 255))
                draw.text((256*(6)+20, 10+64*2), prediction_reward_text, fill=(255, 255, 255))
                draw.text((256*(6)+20, 10+64*3), actual_reward_text, fill=(255, 255, 255))

                unique_id = unique_ids[i]

                # save the input images
                if prediction_action == label_action and prediction_reward == label_reward:
                    os.makedirs(os.path.join(right_predictions_dir, reward_strs[label_reward.item()]), exist_ok=True)
                    paste_into_.save(os.path.join(right_predictions_dir, reward_strs[label_reward.item()], f"{unique_id}.png"))
                else:
                    os.makedirs(os.path.join(wrong_predictions_dir, reward_strs[label_reward.item()]), exist_ok=True)
                    paste_into_.save(os.path.join(wrong_predictions_dir, reward_strs[label_reward.item()], f"{unique_id}.png"))

        if take is not None and ctr >= take:
            break
    
    avg_loss = total_loss / ctr
    avg_loss_actions = total_loss_actions / ctr
    avg_loss_rewards = total_loss_rewards / ctr

    accuracy = 100 * correct / total
    accuracy_actions = 100 * correct_actions / total
    accuracy_rewards = 100 * correct_rewards / total    

    model.train()

    rv = dict(
        avg_loss=avg_loss, accuracy=accuracy, 
        avg_loss_actions=avg_loss_actions, accuracy_actions=accuracy_actions, 
        avg_loss_rewards=avg_loss_rewards, accuracy_rewards=accuracy_rewards,
    )

    do_box_plot = False

    if do_box_plot:
        data_lists = [true_pos_reward_expected_reward, true_neutral_reward_expected_reward]
        labels = ['true_pos_reward_expected_reward', 'true_neutral_reward_expected_reward']
        if len(true_neg_reward_expected_reward):
            data_lists.append(true_neg_reward_expected_reward)
            labels.append('true_neg_reward_expected_reward')
        generate_box_plot(data_lists, labels, 'boxplot.png', title='Sample Box Plot')

    # compare positive to neutral
    sample_n = min(min(1000, len(true_pos_reward_expected_reward)), len(true_neutral_reward_expected_reward))
    approx_positive_rew_better_than_neutral = np.mean(np.random.choice(true_pos_reward_expected_reward, sample_n) > np.random.choice(true_neutral_reward_expected_reward, sample_n))
    rv['approx_positive_rew_better_than_neutral'] = approx_positive_rew_better_than_neutral

    if len(true_neg_reward_expected_reward):
        # compare neutral to negative
        sample_n = min(min(1000, len(true_neutral_reward_expected_reward)), len(true_neg_reward_expected_reward))
        approx_neutral_rew_better_than_negative = np.mean(np.random.choice(true_neutral_reward_expected_reward, sample_n) > np.random.choice(true_neg_reward_expected_reward, sample_n))
        # compare positive to negative
        sample_n = min(min(1000, len(true_pos_reward_expected_reward)), len(true_neg_reward_expected_reward))
        approx_positive_rew_better_than_negative = np.mean(np.random.choice(true_pos_reward_expected_reward, sample_n) > np.random.choice(true_neg_reward_expected_reward, sample_n))

        rv['approx_positive_rew_better_than_negative'] = approx_positive_rew_better_than_negative
        rv['approx_neutral_rew_better_than_negative'] = approx_neutral_rew_better_than_negative

    return rv

def data_loader_loop(train_loader, train_loader_non_zero_rewards=None):
    for it in train_loader:
        yield it

    if train_loader_non_zero_rewards is not None:
        for _ in range(8):
            for it in train_loader_non_zero_rewards:
                yield it

def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, train_loader_non_zero_rewards=None):
    steps_per_epoch = len(train_loader)

    for epoch in range(num_epochs):
        model.train()

        total_train_loss = 0
        total_train_loss_actions = 0
        total_train_loss_rewards = 0

        train_loss_ctr = 0

        total_pbar = len(train_loader)
        if train_loader_non_zero_rewards is not None:
            total_pbar += len(train_loader_non_zero_rewards) * 8

        for step, it in tqdm.tqdm(enumerate(data_loader_loop(train_loader, train_loader_non_zero_rewards), 1), total=total_pbar):
            optimizer.zero_grad()

            inputs, actions, rewards = it['state'], it['action'], it['reward']

            inputs, actions, rewards = inputs.to(device), actions.to(device), rewards.to(device)

            pred_actions, pred_rewards = model(inputs)

            loss_actions = criterion(pred_actions, actions)
            loss_rewards = criterion(pred_rewards, rewards)

            loss = loss_actions + loss_rewards

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_train_loss_actions += loss_actions.item()
            total_train_loss_rewards += loss_rewards.item()

            train_loss_ctr += 1
            if step % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Step {step}/{steps_per_epoch}, Train Loss: {loss.item():.4f} grad_norm: {grad_norm.item():.4f}')

        print('**************')
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Avg Train Loss over epoch: {total_train_loss / train_loss_ctr:.4f} ")
        print(f'Avg Train Loss Actions: {total_train_loss_actions / train_loss_ctr:.4f} ')
        print(f'Avg Train Loss Rewards: {total_train_loss_rewards / train_loss_ctr:.4f} ')

        if (epoch+1) % args.eval_every_n_epochs == 0 or epoch == num_epochs - 1:
            train_metrics = evaluate_model(model, train_loader, criterion, device, id=f"{epoch}_train")

            print(f"Train Loss: {train_metrics['avg_loss']:.4f} Train Accuracy: {train_metrics['accuracy']:.2f}%")
            print(f"Train Loss Actions: {train_metrics['avg_loss_actions']:.4f} Train Accuracy Actions: {train_metrics['accuracy_actions']:.2f}%")
            print(f"Train Loss Rewards: {train_metrics['avg_loss_rewards']:.4f} Train Accuracy Rewards: {train_metrics['accuracy_rewards']:.2f}%")

            test_metrics = evaluate_model(model, test_loader, criterion, device, id=f"{epoch}_test")

            print(f"Test Loss: {test_metrics['avg_loss']:.4f} Test Accuracy: {test_metrics['accuracy']:.2f}%")
            print(f"Test Loss Actions: {test_metrics['avg_loss_actions']:.4f} Test Accuracy Actions: {test_metrics['accuracy_actions']:.2f}%")
            print(f"Test Loss Rewards: {test_metrics['avg_loss_rewards']:.4f} Test Accuracy Rewards: {test_metrics['accuracy_rewards']:.2f}%")

        print('**************')
    
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

    train_rewards = torch.stack(train_df.reward.tolist())
    test_rewards = torch.stack(test_df.reward.tolist())

    train_positive_rewards = (train_rewards > 0).sum().item()
    train_negative_rewards = (train_rewards < 0).sum().item()
    train_no_rewards = (train_rewards == 0).sum().item()

    test_positive_rewards = (test_rewards > 0).sum().item()
    test_negative_rewards = (test_rewards < 0).sum().item()
    test_no_rewards = (test_rewards == 0).sum().item()

    print(f"Train positive rewards: {train_positive_rewards} negative rewards: {train_negative_rewards} no rewards: {train_no_rewards}")
    print(f"Test positive rewards: {test_positive_rewards} negative rewards: {test_negative_rewards} no rewards: {test_no_rewards}")

    train_non_zero_rewards = train_positive_rewards + train_negative_rewards
    test_non_zero_rewards = test_positive_rewards + test_negative_rewards
    print(f"Train non zero rewards: {train_non_zero_rewards/(train_non_zero_rewards + test_non_zero_rewards):.2f}%")

    train_dataset = StateActionDataset(train_df)
    test_dataset = StateActionDataset(test_df)
    train_loader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False)

    train_loader_non_zero_rewards = None # DataLoader(StateActionDataset(train_df, only_non_zero_rewards=True), num_workers=0, batch_size=args.batch_size, shuffle=True)

    num_classes = df['action'].max() + 1
    # model = SimpleCNN(context_n_frames=context_n_frames, num_classes=num_classes).to(device).train()
    model = AnotherCNN(context_n_frames=context_n_frames, num_classes=num_classes, num_rewards=3).to(device).train()
    # model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f"action_labeler_final.pt")))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(model, train_loader, test_loader, criterion, optimizer, args.epochs, device, train_loader_non_zero_rewards=train_loader_non_zero_rewards)

    # evaluate_model(model, train_loader, criterion, device, id="final_train")
    # evaluate_model(model, test_loader, criterion, device, id="final_test")

    if args.write_new_dataset_dir is not None:
        write_new_dataset(train_df, test_df, model)

@torch.no_grad()
def write_new_dataset(train_df, test_df, model):
    model.eval()

    ds = Dataset(args.write_new_dataset_dir, disable_rew_counter=True)

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

                    act_logits, rew_logits = model(model_input)

                    act = act_logits.argmax().item()
                    rew = rew_logits[0]

                    if act == row['action']:
                        num_right += 1
                else:
                    # set the logit for the actual reward to a large number
                    rew_logits = torch.zeros(3, dtype=torch.float32, device=args.gpu)
                    clipped_reward = row['reward'].sign().long().item() + 1
                    assert clipped_reward in [0, 1, 2]
                    rew_logits[clipped_reward] = 1000
                    rew = rew_logits

                episodes[f"{dir}_{episode_id}"].append((obs, act, rew, end, trunc))

                pbar.update(1)
                pbar.set_postfix(num_right=num_right)

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
        assert rew.ndim == 2
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

    print(f"Num right: {num_right/len(test_df):.2f}%")

    print(f"len(df): {len(df)} len(ds): {len(ds)}")

    assert len(df) == len(ds)

    ds.is_static = True
    ds.save_to_default_path()

def generate_box_plot(data_lists, labels, filename, title='Box Plot', xlabel='Categories', ylabel='Values'):
    import matplotlib.pyplot as plt
    import numpy as np

    """
    Generate a box plot from multiple lists of floats and save it to a file.
    
    :param data_lists: List of lists containing float values
    :param labels: List of labels for each dataset
    :param filename: Name of the file to save the box plot (include extension, e.g., 'boxplot.png')
    :param title: Title of the box plot (default: 'Box Plot')
    :param xlabel: Label for x-axis (default: 'Categories')
    :param ylabel: Label for y-axis (default: 'Values')
    """
    plt.figure(figsize=(12, 7))
    
    # Create the box plot
    bp = plt.boxplot(data_lists, labels=labels, patch_artist=True)
    
    # Customize the box plot colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(data_lists)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    # Add some random points to show the distribution
    for i, data in enumerate(data_lists):
        x = np.random.normal(i + 1, 0.04, len(data))
        plt.scatter(x, data, alpha=0.3, s=5)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

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
