from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention import Attention
from einops import rearrange
from torch import nn

from feta.globals import get_num_frames, set_num_frames


def inject_cogvideo(model: nn.Module) -> None:
    # register hook to update num frames
    model.register_forward_pre_hook(num_frames_hook, with_kwargs=True)
    # replace attention with feta
    for name, module in model.named_modules():
        if "attn" in name and isinstance(module, Attention):
            module.set_processor(FETACogVideoXAttnProcessor2_0())


def num_frames_hook(module, args, kwargs):
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
        self, attn: Attention, image_hidden_states: torch.Tensor, head_dim: int, batch_size: int
    ) -> torch.Tensor:
        num_frames = get_num_frames()

        query_image = attn.to_q(image_hidden_states)
        key_image = attn.to_k(image_hidden_states)

        query_image = query_image.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_image = key_image.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        spatial_dim = int(query_image.shape[2] / num_frames)

        if attn.norm_q is not None:
            query_image = attn.norm_q(query_image)
        if attn.norm_k is not None:
            key_image = attn.norm_k(key_image)

        spatial_dim = int(query_image.shape[2] / num_frames)

        query_image = rearrange(
            query_image, "B N (T S) C -> (B S) N T C", N=attn.heads, T=num_frames, S=spatial_dim, C=head_dim
        )
        key_image = rearrange(
            key_image, "B N (T S) C -> (B S) N T C", N=attn.heads, T=num_frames, S=spatial_dim, C=head_dim
        )
        scale = head_dim**-0.5
        query_image = query_image * scale
        attn_temp = query_image @ key_image.transpose(-2, -1)  # translate attn to float32
        attn_temp = attn_temp.to(torch.float32)
        attn_temp = attn_temp.softmax(dim=-1)

        num_frames = attn_temp.shape[2]
        # Reshape to [batch_size * num_tokens, num_frames, num_frames]
        attn_temp = attn_temp.reshape(-1, num_frames, num_frames)

        # Create a mask for diagonal elements
        diag_mask = torch.eye(num_frames, device=attn_temp.device).bool()
        diag_mask = diag_mask.unsqueeze(0).expand(attn_temp.shape[0], -1, -1)

        # Zero out diagonal elements
        attn_wo_diag = attn_temp.masked_fill(diag_mask, 0)

        # Calculate mean for each token's attention matrix
        # Number of off-diagonal elements per matrix is n*n - n
        num_off_diag = num_frames * num_frames - num_frames
        mean_scores = attn_wo_diag.sum(dim=(1, 2)) / num_off_diag

        # mean_scores_mean = torch.quantile(mean_scores, 0.5) * num_frames * (1 + 0.0556)
        # mean_scores_mean = torch.quantile(mean_scores, 0.5) * (num_frames + 1)
        # mean_scores_mean = mean_scores.mean() * num_frames * (1 + 0.05)
        mean_scores_mean = mean_scores.mean() * (num_frames + 1)
        mean_scores_mean = mean_scores_mean.clamp(min=1)
        return mean_scores_mean

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # ========== FETA ==========
        image_hidden_states = hidden_states.clone()
        # ========== FETA ==========

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

        # ========== FETA ==========
        feta_scores = self._get_feta_scores(attn, image_hidden_states, head_dim, batch_size)
        # ========== FETA ==========

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
        hidden_states = hidden_states * feta_scores
        # ========== FETA ==========

        return hidden_states, encoder_hidden_states
