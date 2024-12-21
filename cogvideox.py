import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from enhance_a_video import enable_enhance, inject_feta_for_cogvideox, set_enhance_weight

pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16)

pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_slicing()
# pipe.vae.enable_tiling()

# ============ Enhance-A-Video ============
# comment the following if you want to use the original model
inject_feta_for_cogvideox(pipe.transformer)
# enhance_weight can be adjusted for better visual quality
set_enhance_weight(1)
enable_enhance()
# ============ Enhance-A-Video ============

prompt = "A cute happy Corgi playing in park"

video_generate = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    use_dynamic_cfg=True,
    guidance_scale=6.0,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(video_generate, "output.mp4", fps=8)
