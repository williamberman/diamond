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
from vector_quantize_pytorch import VectorQuantize, l2norm

from models.blocks import UNet, Conv3x3, GroupNorm
from torch import Tensor
from PIL import Image

from dataclasses import dataclass

logger = logging.get_logger(__name__)

device = int(os.environ["LOCAL_RANK"])

context_n_frames = 5

def main():
    global step

    dataloader = iter(DataLoader(TorchDataset(), batch_size=32, num_workers=8))

    model = ActionVQVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

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

        dot_prods = []
        for i in range(4):
            a = F.normalize(model.quantizer._codebook.embed[0, 0], dim=-1)
            b = F.normalize(model.quantizer._codebook.embed[0, i], dim=-1)
            dot_prods.append((a @ b).item())

        log_args = {
            "loss": loss.item(),
            "mse_loss": mse_loss.item(),
            "vq_loss": vq_loss.item(),
            "grad_norm": grad_norm.item(),
            "indices_used": min_encoding_indices.unique().tolist(),
            "dot_prods": dot_prods,
        }
        print(f"{step}: {log_args}")

        if (step+1) % 300 == 0:
            print(model.quantizer._codebook.embed)

        if (step+1) % 500 == 0:
            torch.save(model.state_dict(), f"model_{step+1}.pt")
            validation(model, use_hold_out=True)
            print('*****************')
            # validation(model, use_hold_out=False)
            print('*****************')

        step += 1

@torch.no_grad()
def validation(model, use_hold_out, mapping=None, dbg=False, n_batches=None):
    model.eval()

    dataloader = DataLoader(TorchDataset(use_hold_out=use_hold_out, infini_iter=False), batch_size=128, num_workers=0)
    action_mapping_scores = defaultdict(int)
    out_of = 0

    if n_batches is None:
        total = len(dataloader)
    else:
        total = min(n_batches, len(dataloader))

    for ctr, batch in enumerate(tqdm.tqdm(dataloader, total=total)):
        frames = batch["frames"].to(device)
        pred_frames, mse_loss, vq_loss, min_encoding_indices = model(frames)
        loss = mse_loss + vq_loss

        target_actions = batch["actions"].to(device)[:, :-1]
        out_of += target_actions.numel()
        # out_of += target_actions.shape[0]

        if dbg:
            # foo = [Image.fromarray(x).resize((256,256)) for x in pred_frames.mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()]
            # os.makedirs("pred", exist_ok=True)
            # for i, x in enumerate(foo):
            #     x.save(f"pred/{i}.png")

            # bar = [Image.fromarray(x).resize((256,256)) for x in frames[:, -1].mul(0.5).add(0.5).mul(255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()]
            # os.makedirs("inp", exist_ok=True)
            # for i, x in enumerate(bar):
            #     x.save(f"inp/{i}.png")

            actions_to_str = {
                0: "NOOP",
                1: "FIRE",
                2: "RIGHT",
                3: "LEFT",
            }
            for i, (ba, pba) in enumerate(zip(target_actions, min_encoding_indices)):
                pba = pba[-1].item()
                if mapping is not None:
                    pba = actions_to_str[mapping[pba]]
                print(f"{i}: {actions_to_str[ba[-1].item()]} -> {pba}")
                if i > 40:
                    break

            model.train()
            return

        for action_mapping_ctr, action_mapping in enumerate(permutations(range(4))):
            target_actions_ = []

            for bta in target_actions:
                target_actions_.append([])
                for ta in bta:
                    target_actions_[-1].append(action_mapping[ta])

            target_actions_ = torch.tensor(target_actions_, device=device, dtype=torch.long)

            score = (target_actions_ == min_encoding_indices).sum().item()
            # score = (target_actions_[:, -1] == min_encoding_indices[:, -1]).sum().item()
            action_mapping_scores[action_mapping_ctr] += score

        if ctr >= n_batches:
            break

    max_mapping_ctr = None
    max_mapping_score = 0
    for action_mapping_ctr, action_mapping_score in action_mapping_scores.items():
        percent = action_mapping_score / out_of
        print(f"action_mapping_ctr: {action_mapping_ctr}, action_mapping_score: {action_mapping_score}, {percent}")

        if action_mapping_score > max_mapping_score:
            max_mapping_score = action_mapping_score
            max_mapping_ctr = action_mapping_ctr

    print(f"max_mapping_ctr: {max_mapping_ctr}, max_mapping_score: {max_mapping_score}, {max_mapping_score / out_of}")

    best_map = list(permutations(range(4)))[max_mapping_ctr]

    model.train()

    return best_map

class TorchDataset(torch.utils.data.IterableDataset):
    def __init__(self, use_hold_out=False, infini_iter=True):
        self.use_hold_out = use_hold_out
        self.infini_iter = infini_iter

        self.ds = Dataset("/mnt/raid/diamond/action_autoencoder/dataset/Breakout_recordings_100k_correct/")
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
                buffer = []

                for _ in range(10):
                    episode_id = random.choice(self.non_hold_out_episodes)
                    episode = self.ds.load_episode(episode_id)

                    for _ in range(len(episode)-context_n_frames):
                        frame_idx = random.randint(0, len(episode)-context_n_frames-1)
                        frames = episode.obs[frame_idx:frame_idx+context_n_frames]
                        actions = episode.act[frame_idx:frame_idx+context_n_frames]
                        buffer.append(dict(frames=frames, actions=actions))

                random.shuffle(buffer)

                for x in buffer:
                    yield x

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
        # self.encoder= InnerModelEncoder(
        #     InnerModelConfig(
        #         img_channels=3,
        #         num_steps_conditioning=context_n_frames,
        #         cond_channels=None,
        #         # depths=[2,2,2,2,2,2,2],
        #         # channels=[64,64,64,64,64,64,64],
        #         # attn_depths=[0,0,0,0,0,0,0],
        #         depths=[2,2,2,2],
        #         channels=[64,64,64,64],
        #         attn_depths=[0,0,0,0],
        #         # dropout=0.75
        #         dropout=0.0
        #     )
        # )
        self.encoder = Convyconv()

        self.quantizer = VectorQuantize(
            dim=16,
            codebook_size=4,
            codebook_dim=16,
            heads=1,
            separate_codebook_per_head=False,
            decay=0.99,
            eps=1e-5, # XXX
            use_cosine_sim=False,
            layernorm_after_project_in=True,
            # threshold_ema_dead_code=2,
            channel_last=True, # XXX
            accept_image_fmap=False, # XXX
            commitment_weight=1.0,
            stochastic_sample_codes=False,
            ema_update=True,
            # orthogonal_reg_weight=1.0,
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
        encoded_frames = self.encoder(frames)
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
    dropout: float = 0.0

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
        self.unet = UNetFirstHalf(cfg.depths, cfg.channels, cfg.attn_depths, cfg.dropout)

    def forward(self, frames: Tensor) -> Tensor:
        diff1 = frames[:, 1] - frames[:, 0]
        diff2 = frames[:, 2] - frames[:, 1]
        diff3 = frames[:, 3] - frames[:, 2]
        diff4 = frames[:, 4] - frames[:, 3]

        encoder_input = torch.stack([torch.zeros_like(diff1), diff1, diff2, diff3, diff4], dim=1)
        encoder_input = encoder_input.flatten(1, 2)

        obs = encoder_input

        x = self.conv_in(obs)
        x = self.unet(x)

        encoded_frames = x

        encoded_frames = encoded_frames.reshape(encoded_frames.shape[0], context_n_frames-1, 16)

        # foo = encoded_frames[:, 1, :]
        # dot_prods = []
        # for b_idx in range(foo.shape[0]):
        #     a = F.normalize(foo[b_idx], dim=-1)
        #     b = F.normalize(foo[0], dim=-1)
        #     dot_prods.append((a @ b).item())
        # print(f"dot_prods: {dot_prods}")
        # print(sum(dot_prods)/len(dot_prods))

        return encoded_frames

from models.blocks import ResBlocks, Downsample, Upsample

class UNetFirstHalf(nn.Module):
    def __init__(self, depths: List[int], channels: List[int], attn_depths: List[int], dropout=0.0) -> None:
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
                    dropout=dropout,
                )
            )
        self.d_blocks = nn.ModuleList(d_blocks)

        self.mid_blocks = ResBlocks(
            list_in_channels=[channels[-1]] * 2,
            list_out_channels=[channels[-1]] * 2,
            cond_channels=None,
            attn=True,
            dropout=dropout,
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

class Convyconv(nn.Module):
    def __init__(self):
        super(Convyconv, self).__init__()
        
        self.single_conv3x3 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.single_conv5x5 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.single_conv7x7 = nn.Conv2d(3, 16, kernel_size=7, padding=3, bias=False)

        self.multi_conv3x3 = nn.Conv2d(3*context_n_frames, 16, kernel_size=3, padding=1, bias=False)
        self.multi_conv5x5 = nn.Conv2d(3*context_n_frames, 16, kernel_size=5, padding=2, bias=False)
        self.multi_conv7x7 = nn.Conv2d(3*context_n_frames, 16, kernel_size=7, padding=3, bias=False)

        self.diff_conv3x3 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.diff_conv5x5 = nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False)
        self.diff_conv7x7 = nn.Conv2d(3, 16, kernel_size=7, padding=3, bias=False)

        self.multi_diff_conv3x3 = nn.Conv2d(3*(context_n_frames-1), 16, kernel_size=3, padding=1, bias=False)
        self.multi_diff_conv5x5 = nn.Conv2d(3*(context_n_frames-1), 16, kernel_size=5, padding=2, bias=False)
        self.multi_diff_conv7x7 = nn.Conv2d(3*(context_n_frames-1), 16, kernel_size=7, padding=3, bias=False)

        dim = 16*context_n_frames*3 + 16*3 + 16*(context_n_frames-1)*3 + 16*3

        self.norm = nn.BatchNorm1d(dim, affine=False)
        self.out = nn.Linear(dim, (context_n_frames-1)*16)
    
    def forward(self, frames):
        bs, n_frames, c, h, w = frames.shape

        frames = torch.where(frames == -1, torch.zeros_like(frames), frames)

        single_imgs = frames.reshape(-1, c, h, w)

        single_conv3x3 = F.adaptive_max_pool2d(self.single_conv3x3(single_imgs), 1).reshape(bs, -1)
        single_conv5x5 = F.adaptive_max_pool2d(self.single_conv5x5(single_imgs), 1).reshape(bs, -1)
        single_conv7x7 = F.adaptive_max_pool2d(self.single_conv7x7(single_imgs), 1).reshape(bs, -1)

        multi_imgs = frames.reshape(bs, -1, h, w)

        multi_conv3x3 = F.adaptive_max_pool2d(self.multi_conv3x3(multi_imgs), 1).reshape(bs, -1)
        multi_conv5x5 = F.adaptive_max_pool2d(self.multi_conv5x5(multi_imgs), 1).reshape(bs, -1)
        multi_conv7x7 = F.adaptive_max_pool2d(self.multi_conv7x7(multi_imgs), 1).reshape(bs, -1)


        diff_imgs = frames[:, 1:] - frames[:, :-1]

        diff_imgs_single = diff_imgs.reshape(-1, c, h, w)
        diff_conv3x3 = F.adaptive_max_pool2d(self.diff_conv3x3(diff_imgs_single), 1).reshape(bs, -1)
        diff_conv5x5 = F.adaptive_max_pool2d(self.diff_conv5x5(diff_imgs_single), 1).reshape(bs, -1)
        diff_conv7x7 = F.adaptive_max_pool2d(self.diff_conv7x7(diff_imgs_single), 1).reshape(bs, -1)

        diff_imgs_multi = diff_imgs.reshape(bs, -1, h, w)
        diff_multi_conv3x3 = F.adaptive_max_pool2d(self.multi_diff_conv3x3(diff_imgs_multi), 1).reshape(bs, -1)
        diff_multi_conv5x5 = F.adaptive_max_pool2d(self.multi_diff_conv5x5(diff_imgs_multi), 1).reshape(bs, -1)
        diff_multi_conv7x7 = F.adaptive_max_pool2d(self.multi_diff_conv7x7(diff_imgs_multi), 1).reshape(bs, -1)

        x = torch.cat([
            single_conv3x3, single_conv5x5, single_conv7x7,
            multi_conv3x3, multi_conv5x5, multi_conv7x7,
            diff_conv3x3, diff_conv5x5, diff_conv7x7,
            diff_multi_conv3x3, diff_multi_conv5x5, diff_multi_conv7x7,
        ], dim=-1)

        x = self.norm(x)
        x = F.relu(x)

        x = self.out(x)

        if (step+1) % 100 == 0:
            dbg_dot_prod(x)

        x = x.reshape(bs, -1, 16)

        return x

def dbg_dot_prod(foo):
    dot_prods = []
    for b_idx in range(foo.shape[0]):
        a = F.normalize(foo[b_idx], dim=-1)
        b = F.normalize(foo[0], dim=-1)
        dot_prods.append((a @ b).item())
    print(f"dot_prods: {dot_prods}")
    print(sum(dot_prods)/len(dot_prods))

if __name__ == "__main__":
    main()