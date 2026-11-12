---
title: ESCNet Explorer
emoji: 🦎
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 5.31.0
app_file: app_hf.py
pinned: true
license: mit
short_description: Camouflaged Object Detection with Frequency & Bayesian Uncertainty
---

# 🦎 ESCNet Explorer

**Stretching ESCNet: Frequency-Domain Feature Enhancement and Bayesian Uncertainty Estimation for Camouflaged Object Detection**

## Features

- **PVT-v2-B5 Backbone** — Pyramid Vision Transformer for multi-scale feature extraction
- **FFT Frequency Branch** — Extracts high-frequency texture cues for edge refinement
- **Bayesian Uncertainty Head** — Per-pixel aleatoric uncertainty via log-variance prediction
- **Interactive Dashboard** — Upload an image and get instant segmentation, overlay, FFT heatmap, and uncertainty map

## Performance (COD10K Benchmark)

| Metric | Value |
|--------|-------|
| S-measure | 0.860 |
| wFmeasure | 0.282 |
| meanFm | 0.300 |
| meanEm | 0.883 |
| MAE | 0.030 |
| MBA | 0.774 |

## Author

**Prince Kumar Dubey** (230101096)
Under the supervision of **Dr. N Kishorjit Singh**
Department of CSE, IIIT Manipur

## Usage

Upload any image containing a camouflaged object and click **Analyze with ESCNet**. The model will produce:

1. **Predicted Mask** — Binary segmentation of the camouflaged object
2. **Overlay Result** — Green overlay showing detected regions
3. **FFT Frequency Heatmap** — Frequency-domain energy visualization
4. **Bayesian Uncertainty Map** — Per-pixel confidence (red = uncertain)
