import torch
import os
import torch.distributed
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from torch.nn.parallel import DistributedDataParallel as DDP
from smol_utils import n_actions, configure_opt
from torch.distributions import Categorical
import torch.nn.functional as F
from models.actor_critic import compute_lambda_returns
from envs.env import make_atari_env
from torch.optim.lr_scheduler import LambdaLR
from coroutines.env_loop import make_env_loop
from PIL import Image
import imageio
import wandb
class dummy_wandb:
    def init(*args, **kwargs): pass
    def log(*args, **kwargs): pass
wandb = dummy_wandb()

device = int(os.environ['LOCAL_RANK'])

dir = "/mnt/raid/diamond/spaceinvaders/actor_critic"
if device == 0:
    os.makedirs(dir, exist_ok=True)

def main():
    torch.distributed.init_process_group(backend="nccl")

    if device == 0:
        wandb.init(project="actor_critic")

    env = make_atari_env("ALE/SpaceInvaders-v5", 1, device, True, 64, None)

    model = DDP(ActorCritic(ActorCriticConfig(
        lstm_dim=512,
        img_channels=3,
        img_size=64,
        channels=[32,32,64,64],
        down=[1,1,1,1],
        num_actions=n_actions,
    )).to(device), device_ids=[device])
    model.module.setup_training(env, ActorCriticLossConfig(
        backup_every=15,
        gamma=0.985,
        lambda_=0.95,
        weight_value_loss=1.0,
        weight_entropy_loss=0.001,
    ))
    model.train()

    ckpts = sorted([int(x.split('.')[0].split('_')[-1]) for x in os.listdir(dir) if x.endswith(".pt")])
    if len(ckpts) > 0:
        step = ckpts[-1]
        sd = torch.load(os.path.join(dir, f"actor_critic_{step}.pt"), map_location=torch.device(f"cuda:{device}"), weights_only=True)
        model.load_state_dict(sd, strict=True)
    else:
        step = 0

    opt = configure_opt(model, lr=1e-4, weight_decay=0, eps=1e-8)
    lr_sched = LambdaLR(opt, lambda s: 1 if s >= 100 else s / 100)

    while True:
        out = train_step(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        opt.step()
        opt.zero_grad()
        lr_sched.step()

        log_args = dict(
            loss=out["loss"],
            loss_actions=out["loss_actions"],
            loss_entropy=out["loss_entropy"],
            loss_values=out["loss_values"],
        )

        if device == 0:
            print(f"step {step} {log_args}")
            wandb.log(log_args, step=step)

        if device == 0 and (step + 1) % 5_000 == 0:
            torch.save(model.state_dict(), os.path.join(dir, f"actor_critic_{step+1}.pt"))
            validation(model, step+1)

        torch.distributed.barrier()

        step += 1

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

if __name__ == "__main__":
    main()