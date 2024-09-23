from torch import nn
import torch
from typing import Optional, Union, List, Tuple
from transformers.utils import logging
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, LlamaPreTrainedModel, _prepare_4d_causal_attention_mask_with_cache_position, LlamaConfig
from data import Dataset

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
import os
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from itertools import permutations
from collections import defaultdict
import tqdm
from vector_quantize_pytorch import VectorQuantize

from models.blocks import UNet, Conv3x3, GroupNorm
from torch import Tensor

from dataclasses import dataclass

logger = logging.get_logger(__name__)

device = int(os.environ["LOCAL_RANK"])

context_n_frames = 5

def main():
    dataloader = iter(DataLoader(TorchDataset(), batch_size=32, num_workers=0))

    model = ActionVQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    step = 0

    while True:
        batch = next(dataloader)
        frames = batch["frames"].to(device)
        pred_frames, mse_loss, vq_loss, min_encoding_indices = model(frames)
        loss = mse_loss + vq_loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        log_args = {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "vq_loss": vq_loss.item(),
            "grad_norm": grad_norm.item(),
            "indices_used": min_encoding_indices.unique().tolist(),
        }

        print(f"{step}: {log_args}")

        if (step+1) % 500 == 0:
            torch.save(model.state_dict(), f"model_{step+1}.pt")
            validation(model, use_hold_out=True)
            print('*****************')
            # validation(model, use_hold_out=False)
            print('*****************')

        step += 1

@torch.no_grad()
def validation(model, use_hold_out):
    model.eval()

    dataloader = DataLoader(TorchDataset(use_hold_out=use_hold_out, infini_iter=False), batch_size=128, num_workers=0)
    action_mapping_scores = defaultdict(int)
    out_of = 0

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        frames = batch["frames"].to(device)
        pred_frames, mse_loss, vq_loss, min_encoding_indices = model(frames)
        loss = mse_loss + vq_loss

        target_actions = batch["actions"].to(device)[:, :-1]
        out_of += target_actions.numel()

        for action_mapping_ctr, action_mapping in enumerate(permutations(range(4))):
            target_actions_ = []

            for bta in target_actions:
                target_actions_.append([])
                for ta in bta:
                    target_actions_[-1].append(action_mapping[ta])

            target_actions_ = torch.tensor(target_actions_, device=device, dtype=torch.long)

            score = (target_actions_ == min_encoding_indices).sum().item()
            action_mapping_scores[action_mapping_ctr] += score

    max_mapping_ctr = None
    max_mapping_score = 0
    for action_mapping_ctr, action_mapping_score in action_mapping_scores.items():
        percent = action_mapping_score / out_of
        print(f"action_mapping_ctr: {action_mapping_ctr}, action_mapping_score: {action_mapping_score}, {percent}")

        if action_mapping_score > max_mapping_score:
            max_mapping_score = action_mapping_score
            max_mapping_ctr = action_mapping_ctr

    print(f"max_mapping_ctr: {max_mapping_ctr}, max_mapping_score: {max_mapping_score}, {max_mapping_score / out_of}")

    model.train()

class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, use_hold_out=False, infini_iter=True):
        self.use_hold_out = use_hold_out
        self.infini_iter = infini_iter

        self.ds = Dataset("/mnt/raid/diamond/action_autoencoder/dataset/Breakout_recordings_100k_labeled_100/")
        self.ds.load_from_default_path()

        hold_out = 10_000
        hold_out_episodes = []
        hold_out_count = 0

        for episode_idx in range(self.ds.num_episodes):
            hold_out_episodes.append(episode_idx)
            hold_out_count += self.ds.lengths[episode_idx]
            if hold_out_count >= hold_out:
                break

        non_hold_out_episodes = [episode_idx for episode_idx in range(self.ds.num_episodes) if episode_idx not in hold_out_episodes]

        if self.use_hold_out:
            tmp = non_hold_out_episodes
            non_hold_out_episodes = hold_out_episodes
            hold_out_episodes = tmp

        hold_out_count = sum([self.ds.lengths[episode_idx] for episode_idx in hold_out_episodes])
        non_hold_out_count = sum([self.ds.lengths[episode_idx] for episode_idx in non_hold_out_episodes])

        print(f"hold_out_count: {hold_out_count}, non_hold_out_count: {non_hold_out_count}")

        self.non_hold_out_episodes = non_hold_out_episodes

        self.n_steps = 0

        for episode_idx in self.non_hold_out_episodes:
            self.n_steps +=self.ds.lengths[episode_idx] - context_n_frames

    def __len__(self):
        return self.n_steps

    @torch.no_grad()
    def __iter__(self):
        if self.infini_iter:
            while True:
                episode_id = random.choice(self.non_hold_out_episodes)
                episode = self.ds.load_episode(episode_id)

                for _ in range(len(episode)-context_n_frames):
                    frame_idx = random.randint(0, len(episode)-context_n_frames-1)
                    frames = episode.obs[frame_idx:frame_idx+context_n_frames]
                    actions = episode.act[frame_idx:frame_idx+context_n_frames]
                    yield dict(frames=frames, actions=actions)
        else:
            for episode_id in self.non_hold_out_episodes:
                episode = self.ds.load_episode(episode_id)

                for frame_idx in range(len(episode)-context_n_frames):
                    frames = episode.obs[frame_idx:frame_idx+context_n_frames]
                    actions = episode.act[frame_idx:frame_idx+context_n_frames]
                    yield dict(frames=frames, actions=actions)

class ActionVQVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder= InnerModelEncoder(
            InnerModelConfig(
                img_channels=3,
                num_steps_conditioning=context_n_frames,
                cond_channels=None,
                depths=[2,2,2,2,2,2,2],
                channels=[64,64,64,64,64,64,64],
                attn_depths=[0,0,0,0,0,0,0]
            )
        )

        self.quantizer = VectorQuantize(
            dim=16,
            codebook_size=4,
            codebook_dim=16,
            heads=1,
            separate_codebook_per_head=False,
            decay=0.99,
            eps=1e-5, # XXX
            use_cosine_sim=True,
            layernorm_after_project_in=True,
            threshold_ema_dead_code=2,
            channel_last=True, # XXX
            accept_image_fmap=False, # XXX
            commitment_weight=1.0,
            stochastic_sample_codes=True,
            ema_update=True,
        )

        self.decoder = InnerModel(
            InnerModelConfig(
                img_channels=3,
                num_steps_conditioning=context_n_frames-1,
                cond_channels=256,
                depths=[2,2,2,2],
                channels=[64,64,64,64],
                attn_depths=[0,0,0,0],
            )
        )

    def forward(self, frames):
        encoded_frames = self.encoder(frames.flatten(1, 2))
        encoded_frames = encoded_frames.reshape(encoded_frames.shape[0], context_n_frames-1, 16)
        z_q, min_encoding_indices, vq_loss = self.quantizer(encoded_frames)
        z_q_cond = z_q.reshape(z_q.shape[0], -1)
        pred_frames = self.decoder(frames[:, :-1, :, :, :].flatten(1, 2), z_q_cond)
        mse_loss = F.mse_loss(pred_frames, frames[:, -1:, :, :, :].flatten(1, 2))
        return pred_frames, mse_loss, vq_loss, min_encoding_indices

@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None
    use_act_emb: bool = False


class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.num_steps_conditioning * 16, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, obs: Tensor, act_or_act_emb: Tensor) -> Tensor:
        cond = self.cond_proj(act_or_act_emb)
        x = self.conv_in(obs)
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x

class InnerModelEncoder(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.conv_in = Conv3x3((cfg.num_steps_conditioning) * cfg.img_channels, cfg.channels[0])
        self.unet = UNetFirstHalf(cfg.depths, cfg.channels, cfg.attn_depths)

    def forward(self, obs: Tensor) -> Tensor:
        x = self.conv_in(obs)
        x = self.unet(x)
        return x

from models.blocks import ResBlocks, Downsample, Upsample

class UNetFirstHalf(nn.Module):
    def __init__(self, depths: List[int], channels: List[int], attn_depths: List[int]) -> None:
        super().__init__()
        assert len(depths) == len(channels) == len(attn_depths)

        d_blocks, u_blocks = [], []
        for i, n in enumerate(depths):
            c1 = channels[max(0, i - 1)]
            c2 = channels[i]
            d_blocks.append(
                ResBlocks(
                    list_in_channels=[c1] + [c2] * (n - 1),
                    list_out_channels=[c2] * n,
                    cond_channels=None,
                    attn=attn_depths[i],
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=None,
            attn=True,
        )

        downsamples = [nn.Identity()] + [Downsample(c) for c in channels[:-1]]
        self.downsamples = nn.ModuleList(downsamples)

    def forward(self, x: Tensor) -> Tensor:
        d_outputs = []
        for block, down in zip(self.d_blocks, self.downsamples):
            x_down = down(x)
            x, block_outputs = block(x_down, None)
            d_outputs.append((x_down, *block_outputs))

        x, _ = self.mid_blocks(x, None)

        return x

if __name__ == "__main__":
    main()