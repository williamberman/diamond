import torch
from torch import nn
import numpy as np
from torch.distributions.categorical import Categorical
import random
import cv2
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import NoopResetEnv, FireResetEnv, ClipRewardEnv, EpisodicLifeEnv
import pandas as pd
from io import BytesIO
from PIL import Image
import argparse
import os
import hashlib

# python save_trajectories.py --save_dir /mnt/raid/orca_rl/trajectory_samples_2 --num_steps 12800000 --gpu 0

def parse_args():
    parser = argparse.ArgumentParser(description="Space Invaders Simulator with Multiple Trajectory Parquet Output")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--policy_dir", type=str, default='/mnt/raid/orca_rl/ppo_space_invaders/', help="Path to the policy file")
    parser.add_argument("--epsilon", type=float, default=0, help="Epsilon for random action selection")
    parser.add_argument("--save_dir", type=str, default='/mnt/raid/orca_rl/trajectory_samples/', help="Directory to save parquet files")
    parser.add_argument("--num_trajectories", type=int, default=None, help="Number of trajectories to generate and save", required=False)
    parser.add_argument("--num_steps", type=int, default=None, help="Number of steps to generate and save", required=False)
    return parser.parse_args()

def generate_random_hash(length=8):
    return hashlib.md5(os.urandom(32)).hexdigest()[:length]

def state_to_bytes(state):
    pil_image = Image.fromarray(state)
    img_byte_arr = BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def load_random_agent():
    policy = random.choice([x for x in os.listdir(args.policy_dir) if x.endswith('.pt')])
    policy_path = os.path.join(args.policy_dir, policy)
    print(f"Loading policy: {policy_path}")
    agent = Agent()
    agent.load_state_dict(torch.load(policy_path, map_location=device))
    agent.to(device)

    return agent, policy_path

@torch.no_grad()
def simulator_dataset(args):
    agent, policy = load_random_agent()

    env = make_env()

    trajectory_num = 0
    step_num = 0

    while True:
        state, _ = env.reset()
        trajectory_data = []
        reset_ctr = 0

        while reset_ctr < 3:
            if random.random() < args.epsilon:
                action = env.action_space.sample()
            else:
                agent_inp = state
                agent_inp = [cv2.resize(x, (84, 84), interpolation=cv2.INTER_AREA) for x in agent_inp]
                agent_inp = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in agent_inp]
                agent_inp = np.stack(agent_inp, axis=0)
                agent_inp = torch.tensor(agent_inp, dtype=torch.float32).unsqueeze(0).to(device)
                action, _, _, _ = agent.get_action_and_value(agent_inp)
                action = action.item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                reward = -1

            trajectory_data.append({
                "policy": policy,
                "epsilon": args.epsilon,
                "reward_this_period": reward,
                "state": state_to_bytes(state[-1]),
                "action": action,
                "next_state": state_to_bytes(next_state[-1])
            })

            if terminated or truncated:
                state, _ = env.reset()
                reset_ctr += 1

            else:
                state = next_state

        trajectory_num += 1
        step_num += len(trajectory_data)

        if trajectory_num % 100 == 0:
            agent, policy = load_random_agent()

        pd.DataFrame(trajectory_data).to_parquet(os.path.join(args.save_dir, f"{generate_random_hash()}.parquet"))
        print(f"Trajectory saved to {args.save_dir}")

        print(f"{trajectory_num} trajectories, {step_num} steps")

        if args.num_trajectories is not None and trajectory_num >= args.num_trajectories:
            break

        if args.num_steps is not None and step_num >= args.num_steps:
            break

def make_env():
    env_id = "ALE/SpaceInvaders-v5"
    env = gym.make(env_id, frameskip=4)
    env = NoopResetEnv(env, noop_max=30)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.FrameStack(env, 4)
    return env

n_actions = make_env().action_space.n

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

if __name__ == "__main__":
    args = parse_args()
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)
    simulator_dataset(args)