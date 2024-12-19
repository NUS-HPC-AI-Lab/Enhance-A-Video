from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
from einops import rearrange
from torch import nn

from enhance_a_video.enhance import feta_score
from enhance_a_video.globals import get_num_frames, is_enhance_enabled, set_num_frames


def inject_feta_for_hunyuanvideo(model: nn.Module) -> None:
    """
    Inject FETA for HunyuanVideo model.
    1. register hook to update num frames
    2. replace attention processor with feta to weight the attention scores
    """
    # register hook to update num frames
    model.register_forward_pre_hook(num_frames_hook, with_kwargs=True)
    # replace attention with feta
    for name, module in model.named_modules():
        if "attn" in name and isinstance(module, Attention) and "transformer_blocks" in name:
            module.set_processor(FETAHunyuanVideoAttnProcessor2_0())


def num_frames_hook(module, args, kwargs):
    """
    Hook to update the number of frames automatically.
    """
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states = args[0]
    num_frames = hidden_states.shape[2]
    p_t = module.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    set_num_frames(post_patch_num_frames)
    return args, kwargs


class FETAHunyuanVideoAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "HunyuanVideoAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def _get_feta_scores(self, attn, query, key, encoder_hidden_states):
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            img_q, img_k = query[:, :, : -encoder_hidden_states.shape[1]], key[:, :, : -encoder_hidden_states.shape[1]]
        else:
            img_q, img_k = query, key

        num_frames = get_num_frames()
        _, num_heads, ST, head_dim = img_q.shape
        spatial_dim = ST / num_frames
        spatial_dim = int(spatial_dim)

        query_image = rearrange(
            img_q, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
        )
        key_image = rearrange(img_k, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim)

        return feta_score(query_image, key_image, head_dim, num_frames)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

        # 1. QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # 3. Rotational positional embeddings applied to latent stream
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            if attn.add_q_proj is None and encoder_hidden_states is not None:
                query = torch.cat(
                    [
                        apply_rotary_emb(query[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        query[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
                key = torch.cat(
                    [
                        apply_rotary_emb(key[:, :, : -encoder_hidden_states.shape[1]], image_rotary_emb),
                        key[:, :, -encoder_hidden_states.shape[1] :],
                    ],
                    dim=2,
                )
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        # ========== FETA ==========
        if is_enhance_enabled():
            feta_scores = self._get_feta_scores(attn, query, key, encoder_hidden_states)
        # ========== FETA ==========

        # 4. Encoder condition QKV projection and normalization
        if attn.add_q_proj is not None and encoder_hidden_states is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)

            encoder_query = encoder_query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([query, encoder_query], dim=2)
            key = torch.cat([key, encoder_key], dim=2)
            value = torch.cat([value, encoder_value], dim=2)

        # 5. Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        # 6. Output projection
        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : -encoder_hidden_states.shape[1]],
                hidden_states[:, -encoder_hidden_states.shape[1] :],
            )

            if getattr(attn, "to_out", None) is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)

            if getattr(attn, "to_add_out", None) is not None:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # ========== FETA ==========
        if is_enhance_enabled():
            hidden_states = hidden_states * feta_scores
        # ========== FETA ==========

        return hidden_states, encoder_hidden_states
