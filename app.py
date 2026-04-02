"""
FlowSlider – HuggingFace Spaces demo

Training-free continuous image editing via fidelity-steering decomposition.
Supports FLUX.1-dev and Stable Diffusion 3 backbones.
"""

import os
import sys
import random

import gradio as gr
import numpy as np
import spaces
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))

from FlowEdit_utils import resize_image_for_flux
from FlowEdit_slider_utils import FlowEditFLUX_Slider, FlowEditSD3_Slider

# ---------------------------------------------------------------------------
# Model cache
# ---------------------------------------------------------------------------

_loaded: dict = {}

MODEL_DEFAULTS = {
    "FLUX.1-dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "T_steps": 28,
        "n_max": 20,
        "src_cfg": 3.5,
        "tar_cfg": 3.5,
    },
    "Stable Diffusion 3": {
        "model_id": "stabilityai/stable-diffusion-3-medium-diffusers",
        "T_steps": 50,
        "n_max": 33,
        "src_cfg": 3.5,
        "tar_cfg": 13.5,
    },
}


def _load_pipe(model_name: str):
    if model_name in _loaded:
        return _loaded[model_name]

    cfg = MODEL_DEFAULTS[model_name]
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if model_name == "FLUX.1-dev":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(cfg["model_id"], torch_dtype=dtype)
    else:
        from diffusers import StableDiffusion3Pipeline
        pipe = StableDiffusion3Pipeline.from_pretrained(cfg["model_id"], torch_dtype=dtype)

    pipe = pipe.to(device)
    _loaded[model_name] = (pipe, pipe.scheduler)
    return pipe, pipe.scheduler


# ---------------------------------------------------------------------------
# Encode / decode helpers
# ---------------------------------------------------------------------------

def _encode_image(pipe, image: Image.Image, device: str):
    tensor = pipe.image_processor.preprocess(image).to(device, dtype=torch.float16)
    with torch.autocast(device), torch.inference_mode():
        x0_denorm = pipe.vae.encode(tensor).latent_dist.mode()
    x0 = (x0_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return x0.to(device)


def _decode_latent(pipe, x0_tar, device: str) -> Image.Image:
    x0_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast(device), torch.inference_mode():
        decoded = pipe.vae.decode(x0_denorm, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded)[0]


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@spaces.GPU
def run_edit(
    model_name: str,
    image: Image.Image,
    src_prompt: str,
    tar_prompt: str,
    strengths_str: str,
    T_steps: int,
    n_max: int,
    src_cfg: float,
    tar_cfg: float,
    seed: int,
    progress=gr.Progress(track_tqdm=True),
):
    if image is None:
        raise gr.Error("Please upload an input image.")
    if not src_prompt.strip():
        raise gr.Error("Source prompt cannot be empty.")
    if not tar_prompt.strip():
        raise gr.Error("Target prompt cannot be empty.")

    try:
        strengths = [float(s.strip()) for s in strengths_str.split(",") if s.strip()]
    except ValueError:
        raise gr.Error("Strengths must be comma-separated numbers, e.g. '1, 2, 3'.")
    if not strengths:
        raise gr.Error("Enter at least one strength value.")

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress(0, desc="Loading model…")
    pipe, scheduler = _load_pipe(model_name)

    image_rgb = image.convert("RGB")
    image_rgb, _ = resize_image_for_flux(image_rgb, max_short_edge=512)

    progress(0.05, desc="Encoding source image…")
    x0_src = _encode_image(pipe, image_rgb, device)

    # In FlowSlider, tar_prompt_neg = src_prompt (fidelity anchor)
    slider_fn = FlowEditFLUX_Slider if model_name == "FLUX.1-dev" else FlowEditSD3_Slider

    gallery: list[tuple[Image.Image, str]] = [(image_rgb, "Original")]

    for idx, s in enumerate(strengths):
        progress(
            (idx + 1) / (len(strengths) + 1),
            desc=f"Generating strength s={s:.1f}  ({idx + 1}/{len(strengths)})…",
        )
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x0_tar = slider_fn(
            pipe=pipe,
            scheduler=scheduler,
            x_src=x0_src,
            src_prompt=src_prompt,
            tar_prompt_pos=tar_prompt,
            tar_prompt_neg=src_prompt,   # fidelity anchor = source prompt
            scale=s,
            T_steps=int(T_steps),
            n_avg=1,
            src_guidance_scale=float(src_cfg),
            tar_guidance_scale=float(tar_cfg),
            n_min=0,
            n_max=int(n_max),
            normalize_v_dir=False,
        )

        edited = _decode_latent(pipe, x0_tar, device)
        gallery.append((edited, f"s = {s:.1f}"))

    return gallery


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _update_defaults(model_name: str):
    cfg = MODEL_DEFAULTS[model_name]
    return (
        gr.update(value=cfg["T_steps"]),
        gr.update(value=cfg["n_max"]),
        gr.update(value=cfg["src_cfg"]),
        gr.update(value=cfg["tar_cfg"]),
    )


# ---------------------------------------------------------------------------
# Pre-computed showcase data
# ---------------------------------------------------------------------------

SHOWCASE = [
    {
        "title": "Decay: Wooden Barn",
        "instruction": "Add rust, rot, damage, and collapse to wooden barn",
        "images": [
            ("examples/barn_original.png", "Original"),
            ("examples/barn_s1.png",       "s = 1"),
            ("examples/barn_s2.png",       "s = 2"),
            ("examples/barn_s3.png",       "s = 3"),
            ("examples/barn_s4.png",       "s = 4"),
            ("examples/barn_s5.png",       "s = 5"),
        ],
    },
    {
        "title": "Color: Sunflower",
        "instruction": "Change the sunflower color from yellow to green",
        "images": [
            ("examples/sunflower_original.png", "Original"),
            ("examples/sunflower_s1.png",       "s = 1"),
            ("examples/sunflower_s2.png",       "s = 2"),
            ("examples/sunflower_s3.png",       "s = 3"),
            ("examples/sunflower_s4.png",       "s = 4"),
            ("examples/sunflower_s5.png",       "s = 5"),
        ],
    },
    {
        "title": "Weather: Railroad",
        "instruction": "Change the weather from cloudy/foggy to clear sunny day",
        "images": [
            ("examples/railroad_original.png", "Original"),
            ("examples/railroad_s1.png",       "s = 1"),
            ("examples/railroad_s2.png",       "s = 2"),
            ("examples/railroad_s3.png",       "s = 3"),
            ("examples/railroad_s4.png",       "s = 4"),
            ("examples/railroad_s5.png",       "s = 5"),
        ],
    },
]

# ---------------------------------------------------------------------------
# Build interface
# ---------------------------------------------------------------------------

DESCRIPTION = """
# FlowSlider — Training-Free Continuous Image Editing

**FlowSlider** enables slider-style control over edit intensity by decomposing
the [FlowEdit](https://arxiv.org/abs/2412.08629) update into two orthogonal components:

| Component | Role | Formula |
|-----------|------|---------|
| **Fidelity term** $V_\\text{fid}$ | Keeps the image close to the source | $V(z^\\text{tar}, t, c_\\text{src}) - V(z^\\text{src}, t, c_\\text{src})$ |
| **Steering term** $V_\\text{steer}$ | Drives the semantic edit toward the target | $V(z^\\text{tar}, t, c_\\text{tar}) - V(z^\\text{tar}, t, c_\\text{src})$ |

The combined update is $V^\\Delta_s = V_\\text{fid} + s \\cdot V_\\text{steer}$, where **$s$ is the edit strength**:

- **$s = 1$** — identical to standard FlowEdit (full edit)
- **$s < 1$** — attenuated edit (subtle change)
- **$s > 1$** — amplified edit (exaggerated change)

Because $V_\\text{fid}$ and $V_\\text{steer}$ are nearly orthogonal, scaling $s$ modulates
semantic change while leaving source fidelity largely intact — unlike naive scaling which
amplifies both components and introduces artifacts.
"""

with gr.Blocks(title="FlowSlider", theme=gr.themes.Soft()) as demo:

    gr.Markdown(DESCRIPTION)

    # ---- Showcase gallery ----
    gr.Markdown("## Pre-computed Examples")
    gr.Markdown("Each strip shows the original image followed by FlowSlider outputs at strengths s = 1 → 5.")

    for ex in SHOWCASE:
        gr.Markdown(f"**{ex['title']}** — *{ex['instruction']}*")
        gr.Gallery(
            value=ex["images"],
            columns=6,
            height=200,
            object_fit="cover",
            show_label=False,
            show_share_button=False,
        )

    gr.Markdown("---")
    gr.Markdown("## Try It Yourself")

    with gr.Row():
        # ── Inputs ──────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            model_name = gr.Radio(
                choices=list(MODEL_DEFAULTS.keys()),
                value="FLUX.1-dev",
                label="Backbone Model",
                info="FLUX.1-dev is recommended. Switching reloads model weights on first use.",
            )
            image_input = gr.Image(type="pil", label="Source Image")

            gr.Markdown("### Prompts")
            src_prompt = gr.Textbox(
                label="Source Prompt  (describe the original image)",
                placeholder="e.g. a wooden barn with shiny metal roof in a grassy field",
                lines=2,
            )
            tar_prompt = gr.Textbox(
                label="Target Prompt  (describe the desired edit)",
                placeholder="e.g. a wooden barn with rusted collapsed roof in an overgrown field",
                lines=2,
            )

            strengths_input = gr.Textbox(
                label="Edit Strengths  (s)",
                value="1, 2, 3",
                info="Comma-separated. s=1 equals standard FlowEdit. s>1 amplifies the edit.",
            )

            with gr.Accordion("Advanced Parameters", open=False):
                T_steps = gr.Slider(minimum=10, maximum=100, step=1, value=28,
                                    label="T steps",
                                    info="Total diffusion timesteps (FLUX: 28, SD3: 50).")
                n_max = gr.Slider(minimum=1, maximum=60, step=1, value=20,
                                  label="n_max",
                                  info="Steps using flow-editing; remainder uses standard sampling.")
                src_cfg = gr.Slider(minimum=1.0, maximum=10.0, step=0.5, value=3.5,
                                    label="Source Guidance Scale")
                tar_cfg = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=3.5,
                                    label="Target Guidance Scale")
                seed = gr.Number(value=42, label="Seed", precision=0)

            run_btn = gr.Button("Generate", variant="primary")

        # ── Outputs ─────────────────────────────────────────────────────────
        with gr.Column(scale=1):
            gallery_out = gr.Gallery(
                label="Results  (original + one image per strength value)",
                columns=4,
                height="auto",
                object_fit="contain",
            )

    model_name.change(
        fn=_update_defaults,
        inputs=[model_name],
        outputs=[T_steps, n_max, src_cfg, tar_cfg],
    )

    run_btn.click(
        fn=run_edit,
        inputs=[
            model_name, image_input,
            src_prompt, tar_prompt,
            strengths_input,
            T_steps, n_max, src_cfg, tar_cfg, seed,
        ],
        outputs=[gallery_out],
    )

    # Quick-load examples (image + prompts)
    gr.Examples(
        examples=[
            [
                "FLUX.1-dev",
                "examples/barn_original.png",
                "Wooden barn. Vertical brown planks, shiny metal roof, fresh painted doors. In grassy field under blue sky.",
                "Wooden barn. Rotted gray planks, rusted collapsed roof, peeling broken doors. In overgrown field under blue sky.",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
            [
                "FLUX.1-dev",
                "examples/sunflower_original.png",
                "a honeybee with yellow and black stripes collecting pollen on a yellow sunflower, macro photography, dark blurred background",
                "a honeybee with yellow and black stripes collecting pollen on a green sunflower, macro photography, dark blurred background",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
            [
                "FLUX.1-dev",
                "examples/railroad_original.png",
                "railroad tracks curving through green grassland under dramatic cloudy sky with fog",
                "railroad tracks curving through green grassland under clear sunny blue sky",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
        ],
        inputs=[
            model_name, image_input,
            src_prompt, tar_prompt,
            strengths_input, T_steps, n_max, src_cfg, tar_cfg, seed,
        ],
        outputs=[gallery_out],
        fn=run_edit,
        label="Load an example",
        cache_examples=False,
    )

    gr.Markdown("""
---
**Paper:** *FlowSlider: Training-Free Continuous Image Editing via Fidelity-Steering Decomposition*
**Backbones:** [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) · [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
""")

if __name__ == "__main__":
    demo.launch()
