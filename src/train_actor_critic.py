import torch
import os
import torch.distributed
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from smol_utils import n_actions, configure_opt
from torch.distributions import Categorical
import torch.nn.functional as F
from models.actor_critic import compute_lambda_returns
from torch.optim.lr_scheduler import LambdaLR
from coroutines.env_loop import make_env_loop
from PIL import Image
import imageio
from envs import WorldModelEnv, make_atari_env, WorldModelEnvConfig
from models.diffusion import DiffusionSamplerConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from models.diffusion.denoiser import Denoiser, DenoiserConfig, InnerModelConfig
from models.diffusion import SigmaDistributionConfig
from data import collate_segments_to_batch, Dataset
from functools import partial
from pathlib import Path
from torch.utils.data import DataLoader
from data import BatchSampler
from coroutines.collector import make_collector, NumToCollect
import time

import wandb
class dummy_wandb:
    def init(*args, **kwargs): pass
    def log(*args, **kwargs): pass
wandb = dummy_wandb()

device = int(os.environ['LOCAL_RANK'])

dir = "/mnt/raid/diamond/spaceinvaders/actor_critic_trained_on_world_model"
if device == 0:
    os.makedirs(dir, exist_ok=True)

train_on_world_model = True

def main():
    torch.distributed.init_process_group(backend="nccl")

    if device == 0:
        wandb.init(project="actor_critic")

    actor_critic = DDP(ActorCritic(ActorCriticConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        channels=[32,32,64,64],
        down=[1,1,1,1],
        num_actions=n_actions,
    )).to(device), device_ids=[device])
    actor_critic.train()

    ckpts = sorted([int(x.split('.')[0].split('_')[-1]) for x in os.listdir(dir) if x.endswith(".pt")])
    if len(ckpts) > 0:
        step = ckpts[-1]
        sd = torch.load(os.path.join(dir, f"actor_critic_{step}.pt"), map_location=torch.device(f"cuda:{device}"), weights_only=True)
        actor_critic.load_state_dict(sd, strict=True)
    else:
        step = 0

    opt = configure_opt(actor_critic, lr=1e-4, weight_decay=0, eps=1e-8)
    lr_sched = LambdaLR(opt, lambda s: 1 if s >= 100 else s / 100)

    if train_on_world_model:
        env = make_world_model_env(actor_critic.module)
    else:
        env = make_real_env()

    actor_critic.module.setup_training(env, ActorCriticLossConfig(
        backup_every=15,
        gamma=0.985,
        lambda_=0.95,
        weight_value_loss=1.0,
        weight_entropy_loss=0.001,
    ))

    while True:
        t0 = time.perf_counter()

        out = train_step(actor_critic)
        grad_norm = torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 100.0)
        opt.step()
        opt.zero_grad()
        lr_sched.step()

        log_args = dict(
            loss=out["loss"],
            loss_actions=out["loss_actions"],
            loss_entropy=out["loss_entropy"],
            loss_values=out["loss_values"],
            grad_norm=grad_norm.item(),
            lr=lr_sched.get_last_lr()[0],
            seconds_per_step=time.perf_counter() - t0,
            steps_per_second=1 / (time.perf_counter() - t0),
            steps_per_hour=3600 / (time.perf_counter() - t0),
        )

        if device == 0:
            print(f"step {step} {log_args}")
            wandb.log(log_args, step=step)

        if device == 0 and (step + 1) % 5_000 == 0:
            torch.save(actor_critic.state_dict(), os.path.join(dir, f"actor_critic_{step+1}.pt"))
            validation(actor_critic, step+1)

        torch.distributed.barrier()

        step += 1

def make_real_env():
    return make_atari_env("ALE/SpaceInvaders-v5", 1, device, True, 64, None)

def make_world_model_env(actor_critic):
    num_steps_conditioning = 16

    denoiser = Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(
            img_channels=3,
            num_steps_conditioning=num_steps_conditioning, 
            cond_channels=256,
            depths=[2,2,2,2],
            channels=[64, 64, 64, 64],
            attn_depths=[0, 0, 0, 0],
            num_actions=n_actions,
        ),
        sigma_data=0.5,
        sigma_offset_noise=0.3,
    )).to(device)
    denoiser.setup_training(SigmaDistributionConfig(
        loc=-0.4,
        scale=1.2,
        sigma_min=2e-3,
        sigma_max=20,
    ))
    denoiser.eval()

    denoiser_state_dict = torch.load("/mnt/raid/diamond/spaceinvaders/denoiser/denoiser_275000.pt", map_location="cpu", weights_only=True)
    denoiser_state_dict = {k.replace("module.", ""): v for k, v in denoiser_state_dict.items()}
    denoiser.load_state_dict(denoiser_state_dict, strict=True)

    rew_end_model = RewEndModel(RewEndModelConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        cond_channels=128,
        depths=[2,2,2,2],
        channels=[32,32,32,32],
        attn_depths=[0,0,0,0],
        num_actions=n_actions,
    )).to(device)
    rew_end_model.eval()

    rew_end_model_state_dict = torch.load("/mnt/raid/diamond/spaceinvaders/rew_end/rew_end_1000.pt", map_location="cpu", weights_only=True)
    rew_end_model_state_dict = {k.replace("module.", ""): v for k, v in rew_end_model_state_dict.items()}
    rew_end_model.load_state_dict(rew_end_model_state_dict, strict=True)

    train_dataset = Dataset(Path("dataset") / "train", "train_dataset", False, False)
    make_data_loader = partial(
        DataLoader,
        dataset=train_dataset,
        collate_fn=collate_segments_to_batch,
        num_workers=0,
    )
    bs = BatchSampler(train_dataset, 32, num_steps_conditioning, [0.1, 0.1, 0.1, 0.7])
    data_loader = make_data_loader(batch_sampler=bs)

    collector = make_collector(make_real_env(), actor_critic, train_dataset, 0.01)
    collector.send(NumToCollect(steps=10_000))

    env = WorldModelEnv(denoiser, rew_end_model, data_loader, WorldModelEnvConfig(
        horizon=15,
        num_batches_to_preload=256,
        diffusion_sampler=DiffusionSamplerConfig(
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
    ))

    return env

def train_step(model):
    c = model.module.loss_cfg
    _, act, rew, end, trunc, logits_act, val, val_bootstrap, _ = model.module.env_loop.send(c.backup_every)

    d = Categorical(logits=logits_act)
    entropy = d.entropy().mean()

    lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, c.gamma, c.lambda_)

    loss_actions = (-d.log_prob(act) * (lambda_returns - val).detach()).mean()
    loss_values = c.weight_value_loss * F.mse_loss(val, lambda_returns)
    loss_entropy = -c.weight_entropy_loss * entropy

    loss = loss_actions + loss_entropy + loss_values

    loss.backward()

    return dict(loss=loss.item(), loss_actions=loss_actions.item(), loss_entropy=loss_entropy.item(), loss_values=loss_values.item())

@torch.no_grad()
def validation(model, step):
    model = model.module
    model.eval()

    env = make_atari_env("ALE/SpaceInvaders-v5", 1, device, True, 64, None)
    env_loop = make_env_loop(env, model)
    all_obs = []

    death_ctr = 0

    while True:
        obs, act, rew, end, trunc, logits_act, val, val_bootstrap, infos = env_loop.send(100)

        if  (trunc.any() | end.any()).item():
            death_ctr += 1

        obs = obs.squeeze(0)

        game_over = death_ctr >= 3

        if game_over:
            last_frame = (trunc | end).argmax(dim=-1).item()
            obs = obs[:last_frame]

        all_obs.append(obs)

        if game_over:
            break

    all_obs = torch.cat(all_obs, dim=0)
    all_obs = all_obs.permute(0, 2, 3, 1).div(2).add(0.5).mul(255).clamp(0, 255).byte().cpu().numpy()
    all_obs = [Image.fromarray(x).resize((128, 128)) for x in all_obs]
    imageio.mimsave(os.path.join(dir, f"agent_ep_{step}.mp4"), all_obs, fps=30)

    model.train()


if __name__ == "__main__":
    main()