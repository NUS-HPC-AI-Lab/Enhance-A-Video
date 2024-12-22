# Enhance-A-Video

[Blog](https://oahzxl.github.io/Enhance_A_Video/) | [Twitter](https://x.com/YangL_7/status/1870116980717695243)

This repository is the official implementation of [Enhance-A-Video: Better Generated Video for Free](https://oahzxl.github.io/Enhance_A_Video/).

## 🎥 Demo
![demo](assets/demo.png)

For more demos, please visit our [blog](https://oahzxl.github.io/Enhance_A_Video/).

## 🔥🔥🔥News
- 2024-12-22: We have [ComfyUI-Hunyuan](https://github.com/kijai/ComfyUI-HunyuanVideoWrapper) Version 🚀!
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
| CogVideoX-2B |          480px720px129f          |       50       |      10GB       |

## 🧱 Inference

Generate videos:

```bash
python cogvideox.py
python hunyuanvideo.py
```
