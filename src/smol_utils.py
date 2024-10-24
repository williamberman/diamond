from envs.atari_preprocessing import AtariPreprocessing
from envs.env import TorchEnv
import gymnasium
from typing import Optional
from torch import nn
from torch.optim import AdamW
from torch.distributions import Categorical
import torch
import numpy as np
import cv2
import random
import os
import time

def make_env():
    # return make_atari_env("BreakoutNoFrameskip-v4", 64, None)
    return make_atari_env("ALE/SpaceInvaders-v5", 64, None)

def make_atari_env(
    id: str,
    size: int,
    max_episode_steps: Optional[int],
) -> TorchEnv:
    def env_fn():
        env = gymnasium.make(
            id,
            full_action_space=False,
            frameskip=1,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
        )
        env = AtariPreprocessing(
            env=env,
            noop_max=30,
            frame_skip=4,
            screen_size=size,
        )
        return env

    env = env_fn()

    return env

n_actions = make_env().action_space.n

def configure_opt(model: nn.Module, lr: float, weight_decay: float, eps: float, *blacklist_module_names: str) -> AdamW:
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTMCell, nn.LSTM)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.GroupNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif "bias" in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=lr, eps=eps)
    return optimizer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def process_will_agent_input(obs):
    assert len(obs) == 4
    obs = [cv2.resize(x, (84, 84), interpolation=cv2.INTER_AREA) for x in obs]
    obs = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in obs]
    obs = np.stack(obs, axis=0)
    obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return obs

class WillAgent(nn.Module):
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

class Dataset(torch.utils.data.IterableDataset):
    @torch.no_grad()
    def __iter__(self):
        env = make_env()
        agent = load_random_agent()
        trajectory_ctr = 0

        trajectory_buffer = []

        while True:
            t0 = time.perf_counter()

            if (trajectory_ctr+1) % 200 == 0:
                agent = load_random_agent()

            obs = [] 
            act = [] 
            rew = []
            end = []

            def get_action():
                if len(obs) < 4:
                    return torch.tensor(random.randint(0, n_actions-1))
                else:
                    return agent.get_action_and_value(process_will_agent_input(obs[-4:]))[0].squeeze(0)

            state, _ = env.reset()

            obs.append(state)
            act.append(get_action())
            rew.append(torch.tensor(0, dtype=torch.long))
            end.append(torch.tensor(0, dtype=torch.long))

            while True:
                next_obs, reward_, terminated, truncated, info = env.step(act[-1])

                obs.append(next_obs)
                act.append(get_action())
                rew.append(torch.tensor(reward_).clamp(-1, 1).to(dtype=torch.long))
                end.append(torch.tensor(int(terminated or truncated), dtype=torch.long))

                if terminated or truncated:
                    break
            
            obs = torch.stack([torch.tensor(o).div(255).mul(2).sub(1).permute(2, 0, 1) for o in obs])
            act = torch.stack(act)
            rew = torch.stack(rew)
            end = torch.stack(end)

            trajectory_buffer.append(dict(obs=obs, act=act, rew=rew, end=end))

            if len(trajectory_buffer) == 10:
                yield trajectory_buffer
                trajectory_buffer.clear()

            trajectory_ctr += 1
            # print(f"Collected {trajectory_ctr} trajectories. trajectory_buffer: {len(trajectory_buffer)} time: {time.perf_counter() - t0} steps in trajectory: {obs.shape[0]}")

def load_random_agent(device="cpu"):
    policy_dir = "/mnt/raid/orca_rl/ppo_space_invaders/"
    policy = random.choice([x for x in os.listdir(policy_dir) if x.endswith('.pt')])
    policy_path = os.path.join(policy_dir, policy)
    print(f"Loading policy: {policy_path}")
    agent = WillAgent().to(device)
    agent.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    agent.eval()
    # print(f"Loaded policy: {policy_path}")
    return agent