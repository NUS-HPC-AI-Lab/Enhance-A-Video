import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel
from diffusers.utils import export_to_video

from enhance_a_video import enable_enhance, inject_feta_for_hunyuanvideo, set_enhance_weight

model_id = "tencent/HunyuanVideo"
transformer = HunyuanVideoTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16, revision="refs/pr/18"
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id, transformer=transformer, revision="refs/pr/18", torch_dtype=torch.bfloat16
)
pipe.to("cuda")
# pipe.enable_sequential_cpu_offload()
pipe.vae.enable_tiling()
# pipe.vae.enable_tiling()

# ============ FETA ============
# comment the following if you want to use the original model
inject_feta_for_hunyuanvideo(pipe.transformer)
set_enhance_weight(4)
enable_enhance()
# ============ FETA ============

prompt = "A focused baseball player stands in the dugout, gripping his bat with determination, wearing a classic white jersey with blue pinstripes and a matching cap. The sunlight casts dramatic shadows across his face, highlighting his intense gaze as he prepares for the game. His hands, wrapped in black batting gloves, firmly hold the bat, showcasing his readiness and anticipation. The background reveals the bustling stadium, with blurred fans and vibrant green field, creating an atmosphere of excitement and competition. As he adjusts his stance, the player's concentration and passion for the sport are palpable, embodying the spirit of baseball."

output = pipe(
    prompt=prompt,
    height=544,
    width=960,
    num_frames=129,
    num_inference_steps=50,
    generator=torch.Generator().manual_seed(42),
).frames[0]

export_to_video(output, "output.mp4", fps=15)
