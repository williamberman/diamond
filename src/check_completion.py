# check if the recordings were collected, look in s3://shvaibackups/diamond/tiny/{game}_recordings_100k

import boto3
import os
import tempfile
import json
from data import Dataset

s3 = boto3.client('s3')

def check_recordings_collected(game):
    bucket_name = 'shvaibackups'
    prefix = f'diamond/tiny/{game}_recordings_100k/'
    # list all objects in the bucket with the prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        return [x['Key'] for x in response['Contents']]
    return []

game_idxes = {
    0: ["Amidar", "Assault"],
    1: ["Asterix", "BankHeist"],
    2: ["BattleZone", "Boxing"],
    3: ["Breakout", "ChopperCommand"],
    4: ["CrazyClimber", "DemonAttack"],
    5: ["Freeway", "Frostbite"],
    6: ["Gopher", "Hero"],
    7: ["Jamesbond", "Kangaroo"],
    8: ["Krull", "KungFuMaster"],
    9: ["PrivateEye", "RoadRunner"],
    10: ["Seaquest", "UpNDown"],
}

games = []
for game_idx in game_idxes:
    games.extend(game_idxes[game_idx])

def check_collections():
    # UpNDown has 0 files
    
    for game in games:
        print('*************')
        print(game)
        files = check_recordings_collected(game)
        print(f"files: {len(files)}")
        # assert os.system(f"aws s3 sync s3://shvaibackups/diamond/tiny/{game}_recordings_100k/ /mnt/raid/diamond/tiny/{game}_recordings_100k/") == 0
        total_dataset_len = []
        for dir in os.listdir(f"/mnt/raid/diamond/tiny/{game}_recordings_100k/"):
            dir = os.path.join(f"/mnt/raid/diamond/tiny/{game}_recordings_100k/", dir)
            dataset = Dataset(dir)
            dataset.load_from_default_path()
            total_dataset_len.append(len(dataset))
        print(f"total_dataset_len: {sum(total_dataset_len)} {total_dataset_len}")
        # for x in files:
        #     print(x)
        print('*************')


# check action labelers

def check_action_labeler_checkpoints(game):
    bucket_name = 'shvaibackups'
    prefix = f'diamond/tiny/{game}_action_labelers_100k_1000/'
    # list all objects in the bucket with the prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        return [x['Key'] for x in response['Contents']]
    return []

def check_action_labeler_dataset(game):
    bucket_name = 'shvaibackups'
    prefix = f'diamond/tiny/{game}_recordings_100k_labeled_1000/'
    # list all objects in the bucket with the prefix
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        return [x['Key'] for x in response['Contents']]
    return []

def check_action_labelers():
    for game in games:
        # print(game)
        checkpoints = check_action_labeler_checkpoints(game)
        dataset = check_action_labeler_dataset(game)
    
        # print(f"checkpoints: {len(checkpoints)}")
        final_checkpoint = [x for x in checkpoints if "final" in x]
        is_final_checkpoint = len(final_checkpoint) == 1
        # if not is_final_checkpoint:
        #     print(f"no final checkpoint for {game}")
        # else:
        #     print(f"final checkpoint for {game}: {final_checkpoint[0]}")
    
        # print(f"dataset: {len(dataset)}")
        # for x in checkpoints:
        #     print(x)
        # for x in dataset:
        #     print(x)
    
        if False: # is_final_checkpoint:
            for file_name in ["test_metrics.json", "train_metrics.json"]:
                for i in [99, 199, 299, 399, 499, 599]:
                    file = f"diamond/tiny/{game}_action_labelers_100k_1000/{i}_{file_name}"
    
                    with tempfile.NamedTemporaryFile() as temp_file:
                        s3.download_file('shvaibackups', file, temp_file.name)
                        with open(temp_file.name, 'r') as f:
                            data = json.load(f)
                
                    accuracy = data['accuracy']
                    accuracy_actions = data['accuracy_actions']
                    approx_positive_rew_better_than_neutral = data['approx_positive_rew_better_than_neutral']
    
                    str = f"{i}_{file_name} accuracy: {accuracy:.4f} accuracy_actions: {accuracy_actions:.4f} approx_positive_rew_better_than_neutral: {approx_positive_rew_better_than_neutral:.4f}"
    
                    if "approx_positive_rew_better_than_negative" in data:
                        approx_positive_rew_better_than_negative = data['approx_positive_rew_better_than_negative']
                        str += f" approx_positive_rew_better_than_negative: {approx_positive_rew_better_than_negative:.4f}"
                    if "approx_neutral_rew_better_than_negative" in data:
                        approx_neutral_rew_better_than_negative = data['approx_neutral_rew_better_than_negative']
                        str += f" approx_neutral_rew_better_than_negative: {approx_neutral_rew_better_than_negative:.4f}"
    
                    print(str)
    
        # print('*************')

        if not is_final_checkpoint:
            print(game)

# check_collections()
check_action_labelers()