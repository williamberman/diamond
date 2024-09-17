from models.diffusion import Denoiser, DenoiserConfig, InnerModelConfig, SigmaDistributionConfig
from models.diffusion.denoiser import ComputeLossOutput, add_dims
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from models.diffusion.diffusion_sampler import DiffusionSampler, DiffusionSamplerConfig
import imageio
from PIL import Image
import numpy as np
import tqdm
import random
from smol_conv_classifier import ImprovedCNN
import torch.multiprocessing as mp
from smol_utils import make_env, n_actions, WillAgent, configure_opt, process_will_agent_input

import wandb
class dummy_wandb:
    def init(*args, **kwargs): pass
    def log(*args, **kwargs): pass
# wandb = dummy_wandb()

device = int(os.environ['LOCAL_RANK'])

dir = "/mnt/raid/diamond/spaceinvaders/denoiser"
if device == 0:
    os.makedirs(dir, exist_ok=True)

use_labeled_actions = False
context_len = 16 # 4 orig, # 8/9 frames is the distance over which the enemies move in space invaders

def main():
    mp.set_start_method("spawn") # we use cuda in the dataloader

    torch.distributed.init_process_group(backend="nccl")

    if device == 0:
        wandb.init(project="diamond_denoiser")

    denoiser = DDP(Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(
            img_channels=3,
            num_steps_conditioning=context_len, 
            cond_channels=256,
            depths=[2,2,2,2],
            channels=[64, 64, 64, 64],
            attn_depths=[0, 0, 0, 0],
            num_actions=n_actions,
        ),
        sigma_data=0.5,
        sigma_offset_noise=0.3,
    )).to(device), device_ids=[device] )
    denoiser.module.setup_training(SigmaDistributionConfig(
        loc=-0.4,
        scale=1.2,
        sigma_min=2e-3,
        sigma_max=20,
    ))
    denoiser.train()

    ckpts = sorted([int(x.split('.')[0].split('_')[-1]) for x in os.listdir(dir) if x.endswith(".pt")])
    if len(ckpts) > 0:
        step = ckpts[-1]
        sd = torch.load(os.path.join(dir, f"denoiser_{step}.pt"), map_location=torch.device(f"cuda:{device}"), weights_only=True)
        denoiser.load_state_dict(sd, strict=True)
    else:
        step = 0

    opt = configure_opt(denoiser, lr=1e-4, weight_decay=1e-2, eps=1e-8)

    lr_sched = LambdaLR(opt, lambda s: 1 if s >= 100 else s / 100)

    data_iterator = main_proc_data_iterator()

    while True:
        t0 = time.perf_counter()

        batch = next(data_iterator)
        loss = train_step(denoiser, batch["obs"], batch["act"])

        grad_norm = torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        lr_sched.step()

        log_args = {
            "loss": loss,
            "grad_norm": grad_norm.item(),
            "lr": lr_sched.get_last_lr()[0],
            "seconds_per_step": time.perf_counter() - t0,
            "steps_per_second": 1 / (time.perf_counter() - t0),
            "steps_per_hour": 3600 / (time.perf_counter() - t0),
        }

        if device == 0:
            print(f"Step {step}: {log_args}")
            if (step+1) % (100) == 0:
                wandb.log(log_args, step=step)

        # if device == 0 and (step+1) % 20_000 == 0:
        if device == 0 and (step+1) % 5_000 == 0:
            # torch.save(denoiser.state_dict(), os.path.join(dir, f"denoiser_{step+1}.pt"))
            vid_frames, vid_paths = validation(denoiser, step+1, 5)
            wandb.log({"video": [wandb.Video(x, format='mp4') for x in vid_paths]}, step=step+1)

        torch.distributed.barrier()

        step += 1

def collate_fn(x): return x

@torch.no_grad()
def main_proc_data_iterator():
    chunk_size = context_len + 4 # they do +2 (4->6)

    data_loader = iter(DataLoader(
        dataset=Dataset(),
        num_workers=2,
        batch_size=1, # not real batch size, see below for real batch size
        collate_fn=collate_fn,
        prefetch_factor=2,
    ))

    def inner():
        while True:
            trajectory_buffer = next(data_loader)
            assert len(trajectory_buffer) == 1
            trajectory_buffer = trajectory_buffer[0]

            yield_indices = []

            for trajectory_idx in range(len(trajectory_buffer)):
                trajectory = trajectory_buffer[trajectory_idx][0]

                for start_frame_idx in range(0, trajectory.shape[0]-chunk_size):
                    end_frame_idx = start_frame_idx+chunk_size
                    yield_indices.append((trajectory_idx, start_frame_idx, end_frame_idx))

            random.shuffle(yield_indices)

            for trajectory_idx, start_frame_idx, end_frame_idx in yield_indices:
                obs, act = trajectory_buffer[trajectory_idx]
                obs = obs[start_frame_idx:end_frame_idx]
                act = act[start_frame_idx:end_frame_idx]
                assert len(obs) == len(act)
                assert len(obs) == chunk_size
                yield dict(obs=obs, act=act)

    inner_iterator = inner()

    while True:
        obs = []
        act = []

        for _ in range(8): # 32 single gpu, 4 for 8 gpus, 8 -> 35 for 7 gpus
            x = next(inner_iterator)
            obs.append(x["obs"].to(device))
            act.append(x['act'].to(device))

        yield dict(obs=obs, act=act)

class Dataset(torch.utils.data.IterableDataset):
    @torch.no_grad()
    def __iter__(self):
        env = make_env()
        agent = load_random_agent()
        trajectory_ctr = 0

        trajectory_buffer = []

        if use_labeled_actions:
            action_labeler = ImprovedCNN(n_actions, 3).to(device)
            action_labeler.load_state_dict(torch.load("smol_conv_classifier_4.pt", map_location="cpu", weights_only=True))
            action_labeler.eval()

            labeler_mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
            labeler_std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

        while True:
            t0 = time.perf_counter()

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
            
            if use_labeled_actions:
                obs_ = []
                next_obs_ = []

                for i in range(len(obs)-1):
                    obs_.append(torch.tensor(obs[i]))
                    next_obs_.append(torch.tensor(obs[i+1]))

                obs_ = torch.stack(obs_).to(device).permute(0, 3, 1, 2).float().div(255).sub(labeler_mean).div(labeler_std)
                next_obs_ = torch.stack(next_obs_).to(device).permute(0, 3, 1, 2).float().div(255).sub(labeler_mean).div(labeler_std)
                deltas = next_obs_ - obs_

                labeled_act = action_labeler(deltas).argmax(dim=1).to('cpu')
                labeled_act = torch.concat([labeled_act, act[-1].unsqueeze(0)])
                assert len(labeled_act) == len(obs)
                act = labeled_act
            else:
                act = torch.stack(act)

            obs = torch.stack([torch.tensor(o).div(255).mul(2).sub(1).permute(2, 0, 1) for o in obs])

            trajectory_buffer.append((obs, act))

            if len(trajectory_buffer) == 10:
                yield trajectory_buffer
                trajectory_buffer.clear()

            trajectory_ctr += 1
            # print(f"Collected {trajectory_ctr} trajectories. trajectory_buffer: {len(trajectory_buffer)} time: {time.perf_counter() - t0} steps in trajectory: {obs.shape[0]}")

def train_step(denoiser, batch_obs, batch_act) -> ComputeLossOutput:
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

@torch.no_grad()
def validation(denoiser, step, n_videos):
    denoiser = denoiser.module

    denoiser.eval()

    agent = load_random_agent()
    
    all_wandb_frames = []
    all_paths = []

    for ctr in range(n_videos):
        env = make_env()
        state, _ = env.reset()
        state = [state]
        act = [get_action(agent, state)]

        n_init_steps = random.randint(context_len, 60)

        for _ in range(n_init_steps):
            next_obs, _, _, _, _ = env.step(act[-1])
            state.append(next_obs)
            act.append(get_action(agent, state))

        state = [torch.tensor(x).div(255).mul(2).sub(1).permute(2, 0, 1) for x in state]
        state = torch.stack(state).unsqueeze(0).to(device)
        act = torch.stack(act).unsqueeze(0).to(device)
            
        sampler = DiffusionSampler(denoiser, DiffusionSamplerConfig(
            num_steps_denoising=3,
            sigma_min=2e-3,
            sigma_max=5.0,
            rho=7,
            order=1,  # 1: Euler, 2: Heun
            s_churn=0.0,  # Amount of stochasticity
            s_tmin=0.0,
            s_tmax=float("inf"),
            s_noise=1.0
        ))

        for _ in tqdm.tqdm(range(30*10)):
            next_state, _ = sampler.sample_next_obs(state[:, -context_len:], act[:,-context_len:])
            next_state = next_state.clamp(-1, 1)
            state = torch.cat([state, next_state.unsqueeze(1)], dim=1)
            act = torch.cat([act, torch.tensor(env.action_space.sample(), device=device)[None, None]], dim=1)

        state = state.squeeze(0)[n_init_steps:]
        state = [x.permute(1, 2, 0).add(1).div(2).mul(255).to(torch.uint8).cpu().numpy() for x in state]
        wandb_frames = np.stack([x.transpose(2, 0, 1) for x in state])
        state = [Image.fromarray(x).resize((128,128)) for x in state]
        path = os.path.join(dir, f"trajectory_{step}_{ctr}.mp4")
        imageio.mimsave(path, state, fps=30)

        all_wandb_frames.append(wandb_frames)
        all_paths.append(path)

    denoiser.train()

    return all_wandb_frames, all_paths

def get_action(agent, obs):
    if len(obs) < 4:
        return torch.tensor(random.randint(0, n_actions-1), device='cpu')
    agent_device = next(agent.parameters()).device
    agent_inp = process_will_agent_input(obs[-4:]).to(agent_device)
    action, _, _, _ = agent.get_action_and_value(agent_inp)
    return action.squeeze(0).to('cpu')

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

if __name__ == "__main__":
    main()