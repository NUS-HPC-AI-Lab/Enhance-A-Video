import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

from enhance_a_video import enable_enhance, inject_enhance_for_wan, set_enhance_weight

model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
vae = AutoencoderKLWan.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
    model_id, vae=vae, torch_dtype=torch.bfloat16
)
pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()

# ============ Enhance-A-Video ============
# comment the following if you want to use the original model
inject_enhance_for_wan(pipe.transformer)
# enhance_weight can be adjusted for better visual quality
set_enhance_weight(4)
enable_enhance()
# ============ Enhance-A-Video ============

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    generator=torch.Generator().manual_seed(1),
).frames[0]

export_to_video(output, "output.mp4", fps=15)
