from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
from models.rew_end_model import RewEndModel, RewEndModelConfig
from smol_utils import configure_opt, Dataset, n_actions

import wandb
class dummy_wandb:
    def init(*args, **kwargs): pass
    def log(*args, **kwargs): pass
# wandb = dummy_wandb()

device = int(os.environ['LOCAL_RANK'])

dir = "/mnt/raid/diamond/spaceinvaders/rew_end"
if device == 0:
    os.makedirs(dir, exist_ok=True)

def main():
    torch.distributed.init_process_group(backend="nccl")

    if device == 0:
        wandb.init(project="diamond_rew_end")

    model = DDP(RewEndModel(RewEndModelConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        cond_channels=128,
        depths=[2,2,2,2],
        channels=[32,32,32,32],
        attn_depths=[0,0,0,0],
        num_actions=n_actions,
    )).to(device), device_ids=[device])
    model.train()

    ckpts = sorted([int(x.split('.')[0].split('_')[-1]) for x in os.listdir(dir) if x.endswith(".pt")])
    if len(ckpts) > 0:
        step = ckpts[-1]
        sd = torch.load(os.path.join(dir, f"rew_end_{step}.pt"), map_location=torch.device(f"cuda:{device}"), weights_only=True)
        model.load_state_dict(sd, strict=True)
    else:
        step = 0

    opt = configure_opt(model, 1e-4, 1e-2, 1e-8)
    lr_sched = LambdaLR(opt, lambda s: 1 if s < 100 else s / 100)

    data_iterator = main_proc_data_iterator()

    while True:
        t0 = time.perf_counter()

        batch = next(data_iterator)
        out = train_step(model, batch["obs"], batch["act"], batch["rew"], batch["end"])

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        opt.zero_grad()
        lr_sched.step()

        log_args = {
            "loss": out["loss"].item(),
            "loss_rew": out["loss_rew"].item(),
            "loss_end": out["loss_end"].item(),
            "prob": out['loss'].neg().exp().item(),
            "prob_rew": out['loss_rew'].neg().exp().item(),
            "prob_end": out['loss_end'].neg().exp().item(),
            "grad_norm": grad_norm.item(),
            "lr": lr_sched.get_last_lr()[0],
            "seconds_per_step": time.perf_counter() - t0,
            "steps_per_second": 1 / (time.perf_counter() - t0),
            "steps_per_hour": 3600 / (time.perf_counter() - t0),
        }

        if device == 0:
            print(f"Step {step}: {log_args}")
            if (step+1) % 100 == 0:
                wandb.log(log_args, step=step)

        if device == 0 and (step+1) % 5_000 == 0:
            torch.save(model.state_dict(), os.path.join(dir, f"rew_end_{step+1}.pt"))
            # validation(model, step+1)

        torch.distributed.barrier()

        step += 1

def collate_fn(x): return x

@torch.no_grad()
def main_proc_data_iterator():
    # 15 + 4 = 19 is sequence fed to the model in training script
    chunk_size = 19

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
                trajectory = trajectory_buffer[trajectory_idx]['obs']

                for start_frame_idx in range(0, trajectory.shape[0]-chunk_size):
                    end_frame_idx = start_frame_idx+chunk_size
                    yield_indices.append((trajectory_idx, start_frame_idx, end_frame_idx))

            random.shuffle(yield_indices)

            for trajectory_idx, start_frame_idx, end_frame_idx in yield_indices:
                obs = trajectory_buffer[trajectory_idx]['obs'][start_frame_idx:end_frame_idx]
                act = trajectory_buffer[trajectory_idx]['act'][start_frame_idx:end_frame_idx]
                rew = trajectory_buffer[trajectory_idx]['rew'][start_frame_idx:end_frame_idx]
                end = trajectory_buffer[trajectory_idx]['end'][start_frame_idx:end_frame_idx]
                assert len(obs) == len(act)
                assert len(obs) == len(rew)
                assert len(obs) == chunk_size
                yield dict(obs=obs, act=act, rew=rew, end=end)

    inner_iterator = inner()

    while True:
        obs = []
        act = []
        rew = []
        end = []

        for _ in range(8): # 32 single gpu, 4 for 8 gpus, 8 -> 35 for 7 gpus
            x = next(inner_iterator)
            obs.append(x["obs"].to(device))
            act.append(x['act'].to(device))
            rew.append(x['rew'].to(device))
            end.append(x['end'].to(device))

        yield dict(obs=obs, act=act, rew=rew, end=end)


def train_step(model, batch_obs, batch_act, batch_rew, batch_end):
    pad_to = max([x.shape[0] for x in batch_obs])

    target_rew = []

    for x in batch_rew:
        x = x.sign().long().add(1)
        x = torch.concat([x, torch.full((pad_to-x.shape[0],), -100, device=device)])
        target_rew.append(x)

    target_rew = torch.stack(target_rew)

    target_end = []

    for x in batch_end:
        x = torch.concat([x, torch.full((pad_to-x.shape[0],), -100, device=device)])
        target_end.append(x)

    target_end = torch.stack(target_end)

    batch_obs = torch.stack([torch.concat([x, torch.zeros(pad_to-x.shape[0], *x.shape[1:], device=device, dtype=x.dtype)]) for x in batch_obs])
    batch_act = torch.stack([torch.concat([x, torch.zeros(pad_to-x.shape[0], *x.shape[1:], device=device, dtype=x.dtype)]) for x in batch_act])
    batch_rew = torch.stack([torch.concat([x, torch.zeros(pad_to-x.shape[0], *x.shape[1:], device=device, dtype=x.dtype)]) for x in batch_rew])
    batch_end = torch.stack([torch.concat([x, torch.zeros(pad_to-x.shape[0], *x.shape[1:], device=device, dtype=x.dtype)]) for x in batch_end])

    obs = batch_obs[:, :-1]
    act = batch_act[:, :-1]
    next_obs = batch_obs[:, 1:]

    logits_rew, logits_end, _ = model(obs, act, next_obs)

    loss_rew = F.cross_entropy(logits_rew.reshape(-1, logits_rew.shape[-1]), target_rew[:, :-1].flatten())
    loss_end = F.cross_entropy(logits_end.reshape(-1, logits_end.shape[-1]), target_end[:, :-1].flatten())
    loss = loss_rew + loss_end
    loss.backward()

    return dict(loss=loss, loss_rew=loss_rew, loss_end=loss_end)

if __name__ == "__main__":
    main()