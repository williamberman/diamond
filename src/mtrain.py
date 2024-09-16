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

device = 0

def make_env():
    return make_atari_env("BreakoutNoFrameskip-v4", 64, None)

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

num_actions = make_env().action_space.n

denoiser_cfg = DenoiserConfig(
    inner_model=InnerModelConfig(
        img_channels=3,
        num_steps_conditioning=4,
        cond_channels=256,
        depths=[2,2,2,2],
        channels=[64, 64, 64, 64],
        attn_depths=[0, 0, 0, 0],
        num_actions=num_actions,
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

def main():
    denoiser = Denoiser(denoiser_cfg).to(device)
    denoiser.setup_training(sigma_distribution_cfg)
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

        print(f"Step {step}: {log_args}")

        step += 1

class Dataset(torch.utils.data.IterableDataset):
    @torch.no_grad()
    def __iter__(self):
        env = make_env()

        while True:
            obs = [] 
            act = [] 

            state, _ = env.reset()

            obs.append(state)
            act.append(torch.tensor(env.action_space.sample()))

            while True:
                next_obs, reward, terminated, truncated, info = env.step(act[-1])

                obs.append(next_obs)
                act.append(torch.tensor(env.action_space.sample()))

                if terminated or truncated:
                    break

            obs = [torch.tensor(o).div(255).mul(2).sub(1).permute(2, 0, 1) for o in obs]
            obs = torch.stack(obs)
            act = torch.stack(act)

            yield dict(obs=obs, act=act)

def compute_loss_denoiser(denoiser, batch_obs, batch_act) -> ComputeLossOutput:
    n = denoiser.cfg.inner_model.num_steps_conditioning
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

        sigma = denoiser.sample_sigma_training(b, denoiser.device)
        _, c_out, c_skip, _ = denoiser._compute_conditioners(sigma)

        offset_noise = denoiser.cfg.sigma_offset_noise * torch.randn(b, c, 1, 1, device=next_obs.device)
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

if __name__ == "__main__":
    main()