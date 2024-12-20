import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision="refs/pr/18"
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, revision="refs/pr/18", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.vae.enable_tiling()
# pipe.enable_sequential_cpu_offload()

output = pipe(
    prompt="A cat walks on the grass, realistic",
    height=544,
    width=960,
    num_frames=129,
    num_inference_steps=30,
).frames[0]
export_to_video(output, "output.mp4", fps=15)
