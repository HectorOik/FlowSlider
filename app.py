"""
FlowSlider – HuggingFace Spaces demo

3-prompt flow-matching image editing with continuous intensity control.
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

# Make sure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from FlowEdit_utils import resize_image_for_flux
from FlowEdit_slider_utils import FlowEditFLUX_Slider, FlowEditSD3_Slider

# ---------------------------------------------------------------------------
# Model cache (one pipeline per backbone)
# ---------------------------------------------------------------------------

_loaded: dict = {}

MODEL_DEFAULTS = {
    "FLUX.1-dev": {
        "model_id": "black-forest-labs/FLUX.1-dev",
        "T_steps": 28,
        "n_max": 20,
        "src_cfg": 3.5,
        "tar_cfg": 5.5,
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
# Core encode / decode helpers
# ---------------------------------------------------------------------------

def _encode_image(pipe, image: Image.Image, device: str):
    """Encode a PIL image to a scaled VAE latent tensor."""
    tensor = pipe.image_processor.preprocess(image).to(device, dtype=torch.float16)
    with torch.autocast(device), torch.inference_mode():
        x0_denorm = pipe.vae.encode(tensor).latent_dist.mode()
    x0 = (x0_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return x0.to(device)


def _decode_latent(pipe, x0_tar, device: str) -> Image.Image:
    """Decode a scaled latent tensor back to a PIL image."""
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
    tar_prompt_pos: str,
    tar_prompt_neg: str,
    scales_str: str,
    T_steps: int,
    n_max: int,
    src_cfg: float,
    tar_cfg: float,
    seed: int,
    normalize_v_dir: bool,
    progress=gr.Progress(track_tqdm=True),
):
    # ---- Validate inputs ----
    if image is None:
        raise gr.Error("Please upload an input image.")
    if not src_prompt.strip():
        raise gr.Error("Source prompt cannot be empty.")
    if not tar_prompt_pos.strip():
        raise gr.Error("Positive target prompt cannot be empty.")

    # Default neg prompt = src prompt (standard 2-prompt mode)
    if not tar_prompt_neg.strip():
        tar_prompt_neg = src_prompt

    # Parse scales
    try:
        scales = [float(s.strip()) for s in scales_str.split(",") if s.strip()]
    except ValueError:
        raise gr.Error("Scales must be comma-separated numbers, e.g. '0.0, 0.5, 1.0'.")
    if not scales:
        raise gr.Error("Enter at least one scale value.")

    # Seed
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    progress(0, desc="Loading model…")
    pipe, scheduler = _load_pipe(model_name)

    # Resize input (short edge ≤ 768 to keep VRAM manageable in the Space)
    image_rgb = image.convert("RGB")
    image_rgb, _ = resize_image_for_flux(image_rgb, max_short_edge=512)

    progress(0.05, desc="Encoding source image…")
    x0_src = _encode_image(pipe, image_rgb, device)

    slider_fn = FlowEditFLUX_Slider if model_name == "FLUX.1-dev" else FlowEditSD3_Slider

    gallery: list[tuple[Image.Image, str]] = [(image_rgb, "Original")]

    for idx, scale in enumerate(scales):
        progress(
            (idx + 1) / (len(scales) + 1),
            desc=f"Editing at scale={scale:.2f}  ({idx + 1}/{len(scales)})…",
        )
        # Reset seed for each scale so results are comparable
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        x0_tar = slider_fn(
            pipe=pipe,
            scheduler=scheduler,
            x_src=x0_src,
            src_prompt=src_prompt,
            tar_prompt_pos=tar_prompt_pos,
            tar_prompt_neg=tar_prompt_neg,
            scale=scale,
            T_steps=int(T_steps),
            n_avg=1,
            src_guidance_scale=float(src_cfg),
            tar_guidance_scale=float(tar_cfg),
            n_min=0,
            n_max=int(n_max),
            normalize_v_dir=normalize_v_dir,
        )

        edited = _decode_latent(pipe, x0_tar, device)
        gallery.append((edited, f"scale = {scale:.2f}"))

    return gallery


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _update_defaults(model_name: str):
    """Reset advanced sliders when the user switches backbone."""
    cfg = MODEL_DEFAULTS[model_name]
    return (
        gr.update(value=cfg["T_steps"]),
        gr.update(value=cfg["n_max"]),
        gr.update(value=cfg["src_cfg"]),
        gr.update(value=cfg["tar_cfg"]),
    )


# ---------------------------------------------------------------------------
# Build the Gradio interface
# ---------------------------------------------------------------------------

DESCRIPTION = """
# FlowSlider – Continuous Image Editing via Flow Matching

**FlowSlider** decomposes a semantic edit into a *direction* and a *scale*:

```
V_direction = V_pos − V_neg          (pure editing direction)
V_delta     = (V_neg − V_src) + scale × V_direction
```

| scale | effect |
|-------|--------|
| 0.0 | no edit (stays near source) |
| 1.0 | full edit (positive target applied) |
| > 1 | exaggerated / over-edit |

**Tip:** Leave *Negative Target Prompt* empty to use the source prompt as the baseline
(equivalent to standard 2-prompt FlowEdit).
"""

EXAMPLES = [
    # [model, image(None), src, pos, neg, scales, T, nmax, src_cfg, tar_cfg, seed, norm]
    [
        "FLUX.1-dev", None,
        "a brown bear walking through a stream",
        "a polar bear walking through a stream",
        "",
        "0.0, 0.5, 1.0",
        28, 20, 3.5, 5.5, 42, False,
    ],
    [
        "FLUX.1-dev", None,
        "a building surrounded by green summer trees",
        "a building surrounded by snow-covered winter trees",
        "",
        "0.0, 0.5, 1.0",
        28, 20, 3.5, 5.5, 42, False,
    ],
    [
        "Stable Diffusion 3", None,
        "a red sports car parked on a street",
        "a blue sports car parked on a street",
        "",
        "0.0, 0.5, 1.0",
        50, 33, 3.5, 13.5, 42, False,
    ],
]

with gr.Blocks(title="FlowSlider", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        # ── Left column: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):
            model_name = gr.Radio(
                choices=list(MODEL_DEFAULTS.keys()),
                value="FLUX.1-dev",
                label="Backbone Model",
                info="FLUX.1-dev is recommended. Switching models reloads weights (~30 s on first use).",
            )
            image_input = gr.Image(type="pil", label="Input Image")

            gr.Markdown("### Prompts")
            src_prompt = gr.Textbox(
                label="Source Prompt",
                placeholder="Describe the original image as accurately as possible.",
                lines=2,
            )
            tar_prompt_pos = gr.Textbox(
                label="Positive Target Prompt",
                placeholder="Describe the desired edited result.",
                lines=2,
            )
            tar_prompt_neg = gr.Textbox(
                label="Negative Target Prompt (optional)",
                placeholder="Leave empty to use the source prompt as baseline.",
                lines=2,
            )

            scales_input = gr.Textbox(
                label="Edit Scales",
                value="0.0, 0.5, 1.0",
                info="Comma-separated values. Try 0.0, 0.25, 0.5, 0.75, 1.0 for a full strip.",
            )

            with gr.Accordion("Advanced Parameters", open=False):
                T_steps = gr.Slider(
                    minimum=10, maximum=100, step=1, value=28,
                    label="T steps",
                    info="Total diffusion steps (FLUX default: 28, SD3 default: 50).",
                )
                n_max = gr.Slider(
                    minimum=1, maximum=60, step=1, value=20,
                    label="n_max",
                    info="Number of steps that use flow-editing (rest uses regular sampling).",
                )
                src_cfg = gr.Slider(
                    minimum=1.0, maximum=10.0, step=0.5, value=3.5,
                    label="Source Guidance Scale",
                )
                tar_cfg = gr.Slider(
                    minimum=1.0, maximum=20.0, step=0.5, value=5.5,
                    label="Target Guidance Scale",
                )
                seed = gr.Number(value=42, label="Seed", precision=0)
                normalize_v_dir = gr.Checkbox(
                    value=False,
                    label="Normalize V_dir",
                    info="Normalise the editing direction vector before scaling "
                         "(helps stabilise edit strength across different CFG settings).",
                )

            run_btn = gr.Button("Generate", variant="primary")

        # ── Right column: outputs ────────────────────────────────────────────
        with gr.Column(scale=1):
            gallery_out = gr.Gallery(
                label="Results  (original + one image per scale)",
                columns=3,
                height="auto",
                object_fit="contain",
            )

    # Wire model selector → update defaults
    model_name.change(
        fn=_update_defaults,
        inputs=[model_name],
        outputs=[T_steps, n_max, src_cfg, tar_cfg],
    )

    run_btn.click(
        fn=run_edit,
        inputs=[
            model_name, image_input,
            src_prompt, tar_prompt_pos, tar_prompt_neg,
            scales_input,
            T_steps, n_max, src_cfg, tar_cfg, seed, normalize_v_dir,
        ],
        outputs=[gallery_out],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[
            model_name, image_input,
            src_prompt, tar_prompt_pos, tar_prompt_neg,
            scales_input,
            T_steps, n_max, src_cfg, tar_cfg, seed, normalize_v_dir,
        ],
        outputs=[gallery_out],
        fn=run_edit,
        label="Example Prompts — upload your own image then click an example to load its prompts",
        cache_examples=False,
    )

    gr.Markdown(
        """
---
**Algorithm:** FlowEdit-Slider — 3-prompt directional decomposition for continuous edit-intensity control.
**Backbones:** [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) · [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
"""
    )

if __name__ == "__main__":
    demo.launch()
