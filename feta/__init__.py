from .feta import feta_score
from .globals import enable_feta, get_feta_weight, get_num_frames, is_feta_enabled, set_feta_weight, set_num_frames
from .models.cogvideox import inject_feta_for_cogvideox
from .models.hunyuanvideo import inject_feta_for_hunyuanvideo

__all__ = [
    "inject_feta_for_cogvideox",
    "inject_feta_for_hunyuanvideo",
    "feta_score",
    "get_num_frames",
    "set_num_frames",
    "get_feta_weight",
    "set_feta_weight",
    "enable_feta",
    "is_feta_enabled",
]
