# Enhance-A-Video

This repository is the official implementation of [Enhance-A-Video: Free Temporal Alignment for Video Enhancement](https://oahzxl.github.io/FETA/).

## News
- 2024-12-20: FETA is now available for [CogVideoX](https://github.com/THUDM/CogVideo) and [HunyuanVideo](https://huggingface.co/THUDM/HunyuanVideo-2b)!

## Getting Started

Install the dependencies:

```bash
conda create -n feta python=3.10
conda activate feta
pip install -r requirements.txt
```

Generate videos:

```bash
python cogvideox.py
python hunyuanvideo.py
```
