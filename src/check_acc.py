import os
import glob
import json

game = "Alien"

def sort_key(file):
    return int(file.split('/')[-1].split('.')[0].split('_')[0])

for game in ["MsPacman", "Alien"]:
    train_files = glob.glob(f"/mnt/raid/diamond/tiny/{game}_action_labelers_100k_1000/*train_metrics.json")
    test_files = glob.glob(f"/mnt/raid/diamond/tiny/{game}_action_labelers_100k_1000/*test_metrics.json")
    
    ff = {"train_files": train_files, "test_files": test_files}
    
    for key, files in ff.items():
        print(f"{game} {key}")

        files = sorted(files, key=sort_key)
        
        for file in files:
            sort_key_ = sort_key(file)
            with open(file, "r") as f:
                data = json.load(f)
                accuracy = data['accuracy']
                accuracy_actions = data['accuracy_actions']
                approx_positive_rew_better_than_neutral = data['approx_positive_rew_better_than_neutral']
                print(f"{sort_key_} accuracy: {accuracy:.4f} accuracy_actions: {accuracy_actions:.4f} approx_positive_rew_better_than_neutral: {approx_positive_rew_better_than_neutral:.4f}")
