from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from einops import rearrange
from torch import nn

from enhance_a_video.enhance import feta_score
from enhance_a_video.globals import get_num_frames, is_enhance_enabled, set_num_frames


def inject_feta_for_cogvideox(model: nn.Module) -> None:
    """
    Inject FETA for CogVideoX model.
    1. register hook to update num frames
    2. replace attention processor with feta to weight the attention scores
    """
    # register hook to update num frames
    model.register_forward_pre_hook(num_frames_hook, with_kwargs=True)
    # replace attention with feta
    for name, module in model.named_modules():
        if "attn" in name and isinstance(module, Attention):
            module.set_processor(FETACogVideoXAttnProcessor2_0())


def num_frames_hook(_, args, kwargs):
    """
    Hook to update the number of frames automatically.
    """
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states = args[0]
    num_frames = hidden_states.shape[1]
    set_num_frames(num_frames)
    return args, kwargs


class FETACogVideoXAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def _get_feta_scores(
        self,
        attn: Attention,
        query: torch.Tensor,
        key: torch.Tensor,
        head_dim: int,
        text_seq_length: int,
    ) -> torch.Tensor:
        num_frames = get_num_frames()
        spatial_dim = int((query.shape[2] - text_seq_length) / num_frames)

        query_image = rearrange(
            query[:, :, text_seq_length:],
            "B N (T S) C -> (B S) N T C",
            N=attn.heads,
            T=num_frames,
            S=spatial_dim,
            C=head_dim,
        )
        key_image = rearrange(
            key[:, :, text_seq_length:],
            "B N (T S) C -> (B S) N T C",
            N=attn.heads,
            T=num_frames,
            S=spatial_dim,
            C=head_dim,
        )
        return feta_score(query_image, key_image, head_dim, num_frames)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # ========== FETA ==========
        if is_enhance_enabled():
            feta_scores = self._get_feta_scores(attn, query, key, head_dim, text_seq_length)
        # ========== FETA ==========

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )

        # ========== FETA ==========
        if is_enhance_enabled():
            hidden_states = hidden_states * feta_scores
        # ========== FETA ==========

        return hidden_states, encoder_hidden_states
