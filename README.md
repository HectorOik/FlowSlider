---
title: FlowSlider
emoji: 🎚️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
license: mit
---

# FlowSlider – Continuous Image Editing via Flow Matching

**FlowSlider** extends [FlowEdit](https://huggingface.co/papers/2412.08629) with a
3-prompt directional decomposition that gives you a continuous **scale** knob over
the edit intensity — from "no change" to "full edit" and beyond.

## Algorithm

```
V_direction = V_pos − V_neg          (pure editing direction)
V_delta     = (V_neg − V_src) + scale × V_direction
```

| scale | effect |
|-------|--------|
| 0.0 | no edit |
| 1.0 | full positive-target edit |
| > 1 | exaggerated / over-edit |

## How to use

1. Upload a source image.
2. Fill in **Source Prompt** (describe the image), **Positive Target Prompt** (desired result), and optionally a **Negative Target Prompt** (the "opposite" direction; leave blank to default to the source prompt).
3. Enter one or more **scales** as comma-separated values (e.g. `0.0, 0.5, 1.0`).
4. Click **Generate** — the gallery shows the original plus one result per scale.

## Supported backbones

| Model | HF Hub |
|-------|--------|
| FLUX.1-dev | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| Stable Diffusion 3 | [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) |
