import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from enhance_a_video import enable_enhance, inject_feta_for_cogvideox, set_enhance_weight

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", torch_dtype=torch.bfloat16)

pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()

# ============ FETA ============
# comment the following if you want to use the original model
inject_feta_for_cogvideox(pipe.transformer)
set_enhance_weight(1)
enable_enhance()
# ============ FETA ============

prompt = "A Japanese tram glides through the snowy streets of a city, its sleek design cutting through the falling snowflakes with grace."

video_generate = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    use_dynamic_cfg=True,
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(video_generate, "output.mp4", fps=8)
