from models.diffusion import Denoiser, DenoiserConfig, InnerModelConfig, SigmaDistributionConfig
from models.diffusion.denoiser import ComputeLossOutput, add_dims
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import gymnasium
from envs.atari_preprocessing import AtariPreprocessing
from envs.env import TorchEnv, DoneOnLifeLoss
from typing import Optional
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
import imageio
from PIL import Image
import numpy as np
import wandb
import tqdm
from torch.distributions import Categorical
import random
import cv2

device = int(os.environ['LOCAL_RANK'])

dir = "/mnt/raid/diamond/spaceinvaders/denoiser"
os.makedirs(dir, exist_ok=True)

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

denoiser_cfg = DenoiserConfig(
    inner_model=InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=4,
        cond_channels=256,
        depths=[2,2,2,2],
        channels=[64, 64, 64, 64],
        attn_depths=[0, 0, 0, 0],
        num_actions=n_actions,
    ),
    sigma_data=0.5,
    sigma_offset_noise=0.3,
)

sigma_distribution_cfg = SigmaDistributionConfig(
    loc=-0.4,
    scale=1.2,
    sigma_min=2e-3,
    sigma_max=20,
)

denoiser_optimizer_cfg = {
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "eps": 1e-8,
}

denoiser_lr_warmup_steps = 100

denoiser_max_grad_norm = 1.0

denoiser_batch_size =  32

diffusion_sampler_cfg = DiffusionSamplerConfig(
    num_steps_denoising=3,
    sigma_min=2e-3,
    sigma_max=5.0,
    rho=7,
    order=1,  # 1: Euler, 2: Heun
    s_churn=0.0,  # Amount of stochasticity
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0
)

def main():
    torch.distributed.init_process_group(backend="nccl")

    if device == 0:
        wandb.init(project="diamond_denoiser")

    denoiser = DDP(Denoiser(denoiser_cfg).to(device), device_ids=[device] )
    denoiser.module.setup_training(sigma_distribution_cfg)
    opt = configure_opt(denoiser, **denoiser_optimizer_cfg)
    lr_sched = LambdaLR(opt, lambda s: 1 if s >= denoiser_lr_warmup_steps else s / denoiser_lr_warmup_steps)
    data_loader = DataLoader(
        dataset=Dataset(),
        num_workers=0,
        batch_size=denoiser_batch_size,
        collate_fn=lambda x: x
    )

    denoiser.train()
    opt.zero_grad()
    data_iterator = iter(data_loader)

    ckpts = sorted([int(x.split('.')[0].split('_')[-1]) for x in os.listdir(dir) if x.endswith(".pt")])
    if len(ckpts) > 0:
        step = ckpts[-1]
        denoiser.load_state_dict(torch.load(os.path.join(dir, f"denoiser_{step}.pt")))
    else:
        step = 0

    while True:
        t0 = time.perf_counter()

        batch = next(data_iterator)
        obs = [x["obs"].to(device) for x in batch]
        act = [x["act"].to(device) for x in batch]
        loss = compute_loss_denoiser(denoiser, obs, act)

        grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), denoiser_max_grad_norm)
        opt.step()
        opt.zero_grad()
        lr_sched.step()

        log_args = {
            "loss": loss,
            "grad_norm": grad_norm.item(),
            "lr": lr_sched.get_last_lr()[0],
            "time": time.perf_counter() - t0,
        }

        if device == 0:
            print(f"Step {step}: {log_args}")
            wandb.log(log_args, step=step)

        if device == 0 and (step+1) % 1000 == 0:
            torch.save(denoiser.state_dict(), os.path.join(dir, f"denoiser_{step+1}.pt"))
            vid, vid_path = sample_trajectory_from_denoiser(denoiser, step+1)
            wandb.log({"video": wandb.Video(vid_path, format='mp4')}, step=step+1)

        torch.distributed.barrier()

        step += 1

@torch.no_grad()
def sample_trajectory_from_denoiser(denoiser, step):
    denoiser = denoiser.module

    env = make_env()
    state, _ = env.reset()
    state = [state]
    act = [env.action_space.sample()]
    # get 4 frames
    for _ in range(3):
        next_obs, _, _, _, _ = env.step(act[-1])
        state.append(next_obs)
        act.append(env.action_space.sample())

    state = [torch.tensor(x).div(255).mul(2).sub(1).permute(2, 0, 1) for x in state]
    state = torch.stack(state).unsqueeze(0).to(device)
    act = torch.stack([torch.tensor(a) for a in act]).unsqueeze(0).to(device)
        
    sampler = DiffusionSampler(denoiser, diffusion_sampler_cfg)

    for _ in tqdm.tqdm(range(30*10)):
        next_state, _ = sampler.sample_next_obs(state[:, -4:], act[:,-4:])
        next_state = next_state.clamp(-1, 1)
        state = torch.cat([state, next_state.unsqueeze(1)], dim=1)
        act = torch.cat([act, torch.tensor(env.action_space.sample(), device=device)[None, None]], dim=1)

    state = state.squeeze(0)
    state = [x.permute(1, 2, 0).add(1).div(2).mul(255).to(torch.uint8).cpu().numpy() for x in state]
    wandb_frames = np.stack([x.transpose(2, 0, 1) for x in state])
    state = [Image.fromarray(x).resize((128,128)) for x in state]
    path = os.path.join(dir, f"trajectory_{step}.mp4")
    imageio.mimsave(path, state, fps=30)
    return wandb_frames, path

def get_action(agent, obs):
    if len(obs) < 4:
        return torch.tensor(random.randint(0, n_actions-1))

    agent_inp = obs[-4:]
    agent_inp = [cv2.resize(x, (84, 84), interpolation=cv2.INTER_AREA) for x in agent_inp]
    agent_inp = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in agent_inp]
    agent_inp = np.stack(agent_inp, axis=0)
    agent_inp = torch.tensor(agent_inp, dtype=torch.float32).unsqueeze(0)
    action, _, _, _ = agent.get_action_and_value(agent_inp)
    return torch.tensor(action.item())

class Dataset(torch.utils.data.IterableDataset):
    @torch.no_grad()
    def __iter__(self):
        env = make_env()
        agent = load_random_agent()
        trajectory_ctr = 0

        while True:
            if (trajectory_ctr+1) % 200 == 0:
                agent = load_random_agent()

            obs = [] 
            act = [] 

            state, _ = env.reset()

            obs.append(state)
            act.append(get_action(agent, obs))

            while True:
                next_obs, reward, terminated, truncated, info = env.step(act[-1])

                obs.append(next_obs)
                act.append(get_action(agent, obs))

                if terminated or truncated:
                    break

            obs = [torch.tensor(o).div(255).mul(2).sub(1).permute(2, 0, 1) for o in obs]
            obs = torch.stack(obs)
            act = torch.stack(act)

            # yield chunks of 6 (what they do)
            for i in range(0, obs.shape[0], 6):
                yield dict(obs=obs[i:i+6], act=act[i:i+6])

            trajectory_ctr += 1

def compute_loss_denoiser(denoiser, batch_obs, batch_act) -> ComputeLossOutput:
    n = denoiser.module.cfg.inner_model.num_steps_conditioning
    _, c, h, w = batch_obs[0].shape

    loop_len = max([obs.shape[0] for obs in batch_obs])-n-1
    total_loss = 0.0

    for start_idx in range(0, loop_len):
        obs = []
        act = []
        next_obs = []
        batch_idxes = []

        for batch_idx in range(len(batch_obs)):
            if start_idx+n < len(batch_obs[batch_idx]):
                obs_ = batch_obs[batch_idx][start_idx:start_idx+n]
                act_ = batch_act[batch_idx][start_idx:start_idx+n]

                next_obs_ = batch_obs[batch_idx][start_idx+n]

                assert obs_.shape[0] == act_.shape[0]
                assert obs_.shape[0] == n

                obs.append(obs_)
                act.append(act_)
                next_obs.append(next_obs_)
                batch_idxes.append(batch_idx)

        obs = torch.stack(obs)
        act = torch.stack(act)
        next_obs = torch.stack(next_obs)

        b, t, c, h, w = obs.shape
        obs = obs.reshape(b, t * c, h, w)

        sigma = denoiser.module.sample_sigma_training(b, denoiser.device)
        _, c_out, c_skip, _ = denoiser.module._compute_conditioners(sigma)

        offset_noise = denoiser.module.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=next_obs.device)
        noisy_next_obs = next_obs + offset_noise + torch.randn_like(next_obs) * add_dims(sigma, next_obs.ndim)

        model_output, denoised = denoiser(noisy_next_obs, sigma, obs, act)

        target = (next_obs - c_skip * noisy_next_obs) / c_out
        loss = F.mse_loss(model_output, target)
        loss = loss.div(loop_len)
        loss.backward()
        total_loss += loss.item()

        denoised = denoised.detach().clamp(-1, 1)

        for obs_idx, batch_idx in enumerate(batch_idxes):
            batch_obs[batch_idx][start_idx+n] = denoised[obs_idx]

    return total_loss

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

def load_random_agent():
    policy_dir = "/mnt/raid/orca_rl/ppo_space_invaders/"
    policy = random.choice([x for x in os.listdir(policy_dir) if x.endswith('.pt')])
    policy_path = os.path.join(policy_dir, policy)
    print(f"Loading policy: {policy_path}")
    agent = Agent()
    agent.load_state_dict(torch.load(policy_path, map_location="cpu"))
    agent.eval()
    print(f"Loaded policy: {policy_path}")
    return agent

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
    main()