from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import Attention
from einops import rearrange
from torch import nn

from enhance_a_video.enhance import enhance_score
from enhance_a_video.globals import get_num_frames, is_enhance_enabled, set_num_frames


def inject_enhance_for_wan(model: nn.Module) -> None:
    """
    Inject enhance score for Wan2.1 model.
    1. register hook to update num frames
    2. replace attention processor with enhance processor to weight the attention scores
    """
    # register hook to update num frames
    model.register_forward_pre_hook(num_frames_hook, with_kwargs=True)
    # replace attention with enhanceAvideo
    for name, module in model.named_modules():
        if "attn1" in name and isinstance(module, Attention):
            module.set_processor(EnhanceWanAttnProcessor2_0())


def num_frames_hook(module, args, kwargs):
    """
    Hook to update the number of frames automatically.
    """
    if "hidden_states" in kwargs:
        hidden_states = kwargs["hidden_states"]
    else:
        hidden_states = args[0]
    num_frames = hidden_states.shape[2]
    p_t = module.config.patch_size[0]
    post_patch_num_frames = num_frames // p_t
    set_num_frames(post_patch_num_frames)
    return args, kwargs


class EnhanceWanAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0."
            )

    def _get_enhance_scores(self, query, key):
        img_q, img_k = query, key

        num_frames = get_num_frames()
        _, num_heads, ST, head_dim = img_q.shape
        spatial_dim = ST / num_frames
        spatial_dim = int(spatial_dim)

        query_image = rearrange(
            img_q, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim
        )
        key_image = rearrange(img_k, "B N (T S) C -> (B S) N T C", T=num_frames, S=spatial_dim, N=num_heads, C=head_dim)

        return enhance_score(query_image, key_image, head_dim, num_frames)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # ========== Enhance-A-Video ==========
        if is_enhance_enabled():
            enhance_scores = self._get_enhance_scores(query, key)
        # ========== Enhance-A-Video ==========

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        # ========== Enhance-A-Video ==========
        if is_enhance_enabled():
            hidden_states = hidden_states * enhance_scores
        # ========== Enhance-A-Video ==========

        return hidden_states
