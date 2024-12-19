from .enhance import feta_score
from .globals import (
    enable_enhance,
    get_enhance_weight,
    get_num_frames,
    is_enhance_enabled,
    set_enhance_weight,
    set_num_frames,
)
from .models.cogvideox import inject_feta_for_cogvideox
from .models.hunyuanvideo import inject_feta_for_hunyuanvideo

__all__ = [
    "inject_feta_for_cogvideox",
    "inject_feta_for_hunyuanvideo",
    "feta_score",
    "get_num_frames",
    "set_num_frames",
    "get_enhance_weight",
    "set_enhance_weight",
    "enable_enhance",
    "is_enhance_enabled",
]
