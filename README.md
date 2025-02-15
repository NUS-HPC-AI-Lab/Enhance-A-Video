# Enhance-A-Video

[Paper](https://arxiv.org/abs/2502.07508) | [Blog](https://oahzxl.github.io/Enhance_A_Video/) | [Twitter](https://x.com/YangL_7/status/1870116980717695243) | [Reddit](https://www.reddit.com/r/StableDiffusion/comments/1hj4f18/enhanceavideo_better_generared_video_for_free/?rdt=46236)

This repository is the official implementation of [Enhance-A-Video: Better Generated Video for Free](https://oahzxl.github.io/Enhance_A_Video/).

## 🎥 Demo
<div align="center">
  <video src="https://github.com/user-attachments/assets/4552d8cf-2f45-49df-8da9-fd74b3ab1368" width="50%">
</div>

The video has been heavily compressed to GitHub's policy. For more demos, please visit our [blog](https://oahzxl.github.io/Enhance_A_Video/).

## 🔥🔥🔥News
- 2025-02-11: Release Enhance-A-Video paper: [Enhance-A-Video: Better Generated Video for Free](https://arxiv.org/abs/2502.07508).
- 2024-12-22: Our work achieves improvements on [LTX-Video](https://github.com/Lightricks/LTX-Video) and has been added to [ComfyUI-LTX](https://github.com/logtd/ComfyUI-LTXTricks). Many thanks to [kijai](https://github.com/kijai) 👏!
- 2024-12-22: Our work is added to [ComfyUI-Hunyuan](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) 🚀!
- 2024-12-20: Enhance-A-Video is now available for [CogVideoX](https://github.com/THUDM/CogVideo) and [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)!
- 2024-12-20: We have released code and [blog](https://oahzxl.github.io/Enhance_A_Video/) for Enhance-A-Video!

## 🎉 Method

![method](assets/method.png)

We design an Enhance Block as a parallel branch. This branch computes the average of non-diagonal elements of temporal attention maps as cross-frame intensity (CFI). An enhanced temperature parameter multiplies the CFI to enhance the temporal attention output.

## 🛠️ Dependencies and Installation

Install the dependencies:

```bash
conda create -n enhanceAvideo python=3.10
conda activate enhanceAvideo
pip install -r requirements.txt
```

## 📜 Requirements
The following table shows the requirements for running HunyuanVideo/CogVideoX model (batch size = 1) to generate videos:

|    Model     | Setting<br/>(height/width/frame) | Denoising step | GPU Peak Memory |
|:------------:|:--------------------------------:|:--------------:|:---------------:|
| HunyuanVideo |         720px1280px129f          |       50       |      60GB       |
| CogVideoX-2B |          480px720px49f           |       50       |      10GB       |

## 🧱 Inference

Generate videos:

```bash
python cogvideox.py
python hunyuanvideo.py
```

## 🔗 BibTeX
```BibTeX
@misc{luo2025enhanceavideobettergeneratedvideo,
      title={Enhance-A-Video: Better Generated Video for Free}, 
      author={Yang Luo and Xuanlei Zhao and Mengzhao Chen and Kaipeng Zhang and Wenqi Shao and Kai Wang and Zhangyang Wang and Yang You},
      year={2025},
      eprint={2502.07508},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.07508}, 
}
```
