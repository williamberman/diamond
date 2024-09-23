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

logger = logging.get_logger(__name__)

device = int(os.environ["LOCAL_RANK"])

context_n_frames = 4
image_seq_len = ((64//4)**2)

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
            validation(model)

        step += 1

@torch.no_grad()
def validation(model):
    model.eval()

    dataloader = DataLoader(TorchDataset(use_hold_out=True, infini_iter=False), batch_size=128, num_workers=0)
    action_mapping_scores = defaultdict(int)
    out_of = 0

    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        frames = batch["frames"].to(device)
        pred_frames, mse_loss, vq_loss, min_encoding_indices = model(frames)
        loss = mse_loss + vq_loss

        target_actions = batch["actions"].to(device)[:, -2]
        min_encoding_indices = min_encoding_indices.squeeze(-1)
        out_of += target_actions.shape[0]

        for action_mapping_ctr, action_mapping in enumerate(permutations(range(4))):
            target_actions_ = torch.tensor([action_mapping[x] for x in target_actions.tolist()], device=device, dtype=torch.long)
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
        hidden_size = 256
        num_attention_heads = 4
        config = dict(
            vocab_size=None,
            hidden_size=hidden_size,
            intermediate_size=hidden_size*4,
            num_hidden_layers=4,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            hidden_act="silu",
            attention_bias=False,
            attention_dropout=0.0,
            mlp_bias=False,
        )
        self.latent_dim = 6

        self.in_proj = nn.Linear(3*(4**2), hidden_size)
        self.encoder = LlamaModel(LlamaConfig(
            **config,
            max_position_embeddings=image_seq_len*(context_n_frames-1),
        ))

        # self.bottleneck = nn.Linear(hidden_size, self.latent_dim)
        # self.quantizer = VectorQuantizer(n_e=4, e_dim=self.latent_dim, beta=0.01)
        # self.quantizer = LFQ(codebook_size=64, entropy_loss_weight=0.25)
        # self.un_bottleneck = nn.Linear(self.latent_dim, hidden_size)

        self.quantizer = VectorQuantize(
            dim=hidden_size,
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

        self.decoder = LlamaModel(LlamaConfig(
            **config,
            max_position_embeddings=image_seq_len*context_n_frames+1,
        ))
        self.out_proj = nn.Linear(hidden_size, 3*(4**2))


    def forward(self, frames):
        target_frames = frames[:, -1, :, :, :]

        frames = frames[:, :-1, :, :, :] # do not pass in the last frame to the encoder
        frames = F.pixel_unshuffle(frames, 4)

        # batch, time, channels, height, width -> batch, seq, inner dim
        frames = frames.permute(0, 1, 3, 4, 2).flatten(1, 3).contiguous()

        frames = self.in_proj(frames)
        input_frames = frames
        frames = self.encoder(inputs_embeds=frames, use_cache=False).last_hidden_state

        last_tokens = frames[:, -1:, :]
        # last_tokens = self.bottleneck(last_tokens)
        # assert last_tokens.shape == (frames.shape[0], 1, self.hidden_size)

        # vq_loss, z_q, perplexity, min_encodings, min_encoding_indices = self.quantizer(last_tokens)
        # ret = self.quantizer(last_tokens)
        # z_q = ret.quantized
        # min_encoding_indices = ret.indices
        # vq_loss = ret.entropy_aux_loss
        # print(min_encoding_indices.unique())

        z_q, min_encoding_indices, vq_loss = self.quantizer(last_tokens)

        # z_q = self.un_bottleneck(z_q)

        frames = torch.concat([
            input_frames, 
            z_q, 
            torch.ones(
                input_frames.shape[0], image_seq_len, input_frames.shape[-1], 
                dtype=input_frames.dtype, device=input_frames.device)
        ], dim=1)

        pred_frames = self.decoder(inputs_embeds=frames, use_cache=False).last_hidden_state
        pred_frames = pred_frames[:, -image_seq_len:, :]
        pred_frames = self.out_proj(pred_frames)
        pred_frames = pred_frames.reshape(pred_frames.shape[0], int(image_seq_len**0.5), int(image_seq_len**0.5), -1)
        pred_frames = pred_frames.permute(0, 3, 1, 2)
        pred_frames = F.pixel_shuffle(pred_frames, 4)

        pred_frames = pred_frames.sigmoid() - 1

        mse_loss = F.mse_loss(pred_frames, target_frames)

        return pred_frames, mse_loss, vq_loss, min_encoding_indices


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


class VectorQuantizerOld(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizerOld, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (batch, channel, height, width)

        quantization pipeline:

            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)

        """
        # reshape z -> (batch, height, width, channel) and flatten
        # z = z.permute(0, 2, 3, 1).contiguous()
        batch, seq, inner_dim = z.shape
        assert inner_dim == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e, device=z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))
        # print(min_encoding_indices.unique())

        # reshape back to match original input shape
        # z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

if __name__ == "__main__":
    main()