import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from feta import enable_feta, inject_feta_for_cogvideox, set_feta_weight

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)

pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()

# ============ FETA ============
inject_feta_for_cogvideox(pipe.transformer)
set_feta_weight(1)
enable_feta()
# ============ FETA ============

prompt = "A Japanese tram glides through the snowy streets of a city, its sleek design cutting through the falling snowflakes with grace."

video_generate = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    use_dynamic_cfg=True,
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(video_generate, "output.mp4", fps=8)
