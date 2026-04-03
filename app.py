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
        "T_steps": 28,
        "n_max": 20,
        "src_cfg": 3.5,
        "tar_cfg": 3.5,
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
    tar_prompt_neg: str,
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

    # Fidelity anchor: use provided negative prompt, or fall back to source prompt
    fidelity_prompt = tar_prompt_neg.strip() if tar_prompt_neg.strip() else src_prompt
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
            tar_prompt=tar_prompt,
            tar_prompt_neg=fidelity_prompt,
            strength=s,
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
        "title": "Decay: Metal mugs",
        "instruction": "Add rust, corrosion, damage, and overgrowth to metal mugs",
        "images": [
            ("examples/mugs_original.png", "Original"),
            ("examples/mugs_s1.png",       "s = 1"),
            ("examples/mugs_s2.png",       "s = 2"),
            ("examples/mugs_s3.png",       "s = 3"),
            ("examples/mugs_s4.png",       "s = 4"),
            ("examples/mugs_s5.png",       "s = 5"),
        ],
    },
    {
        "title": "Season: Summer → Winter",
        "instruction": "Change the season to winter with snow",
        "images": [
            ("examples/tree_winter_original.png", "Original"),
            ("examples/tree_winter_s1.png",       "s = 1"),
            ("examples/tree_winter_s2.png",       "s = 2"),
            ("examples/tree_winter_s3.png",       "s = 3"),
            ("examples/tree_winter_s4.png",       "s = 4"),
            ("examples/tree_winter_s5.png",       "s = 5"),
        ],
    },
    {
        "title": "Season: Autumn → Spring",
        "instruction": "Change the season to spring with fresh green leaves",
        "images": [
            ("examples/leaves_spring_original.png", "Original"),
            ("examples/leaves_spring_s1.png",       "s = 1"),
            ("examples/leaves_spring_s2.png",       "s = 2"),
            ("examples/leaves_spring_s3.png",       "s = 3"),
            ("examples/leaves_spring_s4.png",       "s = 4"),
            ("examples/leaves_spring_s5.png",       "s = 5"),
        ],
    },
    {
        "title": "Season: Spring → Autumn",
        "instruction": "Change the season to autumn with warm fall colors",
        "images": [
            ("examples/lake_autumn_original.png", "Original"),
            ("examples/lake_autumn_s1.png",       "s = 1"),
            ("examples/lake_autumn_s2.png",       "s = 2"),
            ("examples/lake_autumn_s3.png",       "s = 3"),
            ("examples/lake_autumn_s4.png",       "s = 4"),
            ("examples/lake_autumn_s5.png",       "s = 5"),
        ],
    },
    {
        "title": "Time of Day: Overcast → Sunset",
        "instruction": "Change the time to sunset with golden light",
        "images": [
            ("examples/lofoten_sunset_original.png", "Original"),
            ("examples/lofoten_sunset_s1.png",       "s = 1"),
            ("examples/lofoten_sunset_s2.png",       "s = 2"),
            ("examples/lofoten_sunset_s3.png",       "s = 3"),
            ("examples/lofoten_sunset_s4.png",       "s = 4"),
            ("examples/lofoten_sunset_s5.png",       "s = 5"),
        ],
    },
]

# ---------------------------------------------------------------------------
# Build interface
# ---------------------------------------------------------------------------
DESCRIPTION = """
# FlowSlider: Training-Free Continuous Image Editing via Fidelity-Steering Decomposition
Read the paper on arXiv: [FlowSlider Paper](https://arxiv.org/abs/2604.02088)

**FlowSlider** lets you control how much an image edit happens—from subtle changes to dramatic transformations.

## How It Works

The magic is in separating the edit dynamics into two independent parts:

- **Fidelity** — keeps your image looking like the original
- **Steering** — pushes the image toward your target description

By adjusting the strength slider **`s`**, you can amplify the steering effect while keeping the fidelity anchor intact, giving you smooth continuous control over the edit intensity.

**Try it:** Upload an image, describe what you see and what you want to change, then slide to find your perfect level of edit intensity!
"""

with gr.Blocks(title="FlowSlider", theme=gr.themes.Soft(font=gr.themes.GoogleFont("Inter"))) as demo:

    gr.Markdown(DESCRIPTION)

    # ---- Showcase gallery ----
    gr.Markdown("## Examples")
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
    gr.Markdown("⚠️ **Note:** Due to HuggingFace Spaces resource limits, results are resized to 512px on the short edge and may take ~30 seconds to generate.")

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
            tar_prompt_neg = gr.Textbox(
                label="Negative Target Prompt  (optional)",
                placeholder="Leave empty to use the source prompt as the fidelity anchor.",
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
                                    info="Total diffusion timesteps (FLUX: 28, SD3: 28).")
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
            src_prompt, tar_prompt, tar_prompt_neg,
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
                "examples/mugs_original.png",
                "Metal mugs. Shiny silver surface, smooth cylindrical body, clean metal handles. Standing on stone surface in grassy field.",
                "Metal mugs. Heavily rusted surface, severely corroded cylindrical body, broken shattered handles. Standing on crumbling stone surface in overgrown field.",
                "Metal mugs. Shiny silver surface, smooth cylindrical body, clean metal handles. Standing on stone surface in grassy field.",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
            [
                "FLUX.1-dev",
                "examples/tree_winter_original.png",
                "majestic solitary tree on rolling green alpine meadow, starburst sun rays peeking through branches, long shadow on grass, blue sky with scattered white clouds, distant forested mountains, golden hour light",
                "majestic solitary tree on rolling snow-covered white alpine meadow, starburst sun rays peeking through bare branches, long shadow on snow, blue sky with scattered white clouds, distant snow-covered mountains, winter light",
                "majestic solitary tree on rolling green alpine meadow, starburst sun rays peeking through branches, long shadow on grass, blue sky with scattered white clouds, distant forested mountains, golden hour light",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
            [
                "FLUX.1-dev",
                "examples/leaves_spring_original.png",
                "delicate vibrant orange autumn leaves on slender dark tree branches, soft misty foggy background with blurred tree silhouettes, backlit leaves glowing with warm color, artistic nature photography with shallow depth of field, peaceful fall morning atmosphere",
                "delicate vibrant fresh green spring leaves on slender dark tree branches, soft clear bright background with blurred tree silhouettes, backlit leaves glowing with green color, artistic nature photography with shallow depth of field, peaceful spring morning atmosphere",
                "delicate vibrant orange autumn leaves on slender dark tree branches, soft misty foggy background with blurred tree silhouettes, backlit leaves glowing with warm color, artistic nature photography with shallow depth of field, peaceful fall morning atmosphere",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
            [
                "FLUX.1-dev",
                "examples/lake_autumn_original.png",
                "serene early spring lake with mirror-like reflections of shoreline scenery, manicured green grass meadow with scattered deciduous trees showing fresh pale green and yellow buds, mixed evergreen and bare deciduous forest on rolling hillside, small stone bridge over inlet, moody dramatic cloudy sky with bright breaks, German or Austrian countryside park atmosphere",
                "serene autumn lake with mirror-like reflections of shoreline scenery, golden brown grass meadow with scattered deciduous trees showing vibrant orange and red fall foliage, mixed evergreen and colorful autumn forest on rolling hillside, small stone bridge over inlet, moody dramatic cloudy sky, German or Austrian countryside park autumn atmosphere",
                "serene early spring lake with mirror-like reflections of shoreline scenery, manicured green grass meadow with scattered deciduous trees showing fresh pale green and yellow buds, mixed evergreen and bare deciduous forest on rolling hillside, small stone bridge over inlet, moody dramatic cloudy sky with bright breaks, German or Austrian countryside park atmosphere",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
            [
                "FLUX.1-dev",
                "examples/lofoten_sunset_original.png",
                "breathtaking aerial view from Reinebringen mountain summit overlooking iconic Reine fishing village on Lofoten Islands Norway, dramatic jagged granite mountain peaks rising from deep blue Norwegian Sea fjords, small dark tarn lake in foreground, traditional red and white fishing huts scattered along coastline, winding roads and bridges connecting islands, moody overcast sky with warm light on horizon, spectacular Nordic Arctic archipelago landscape",
                "breathtaking aerial view from Reinebringen mountain summit overlooking iconic Reine fishing village on Lofoten Islands Norway, dramatic jagged granite mountain peaks rising from golden reflective Norwegian Sea fjords, small dark tarn lake in foreground, traditional red and white fishing huts scattered along coastline, winding roads and bridges connecting islands, dramatic warm orange and pink sunset sky, spectacular Nordic Arctic archipelago golden hour landscape",
                "breathtaking aerial view from Reinebringen mountain summit overlooking iconic Reine fishing village on Lofoten Islands Norway, dramatic jagged granite mountain peaks rising from deep blue Norwegian Sea fjords, small dark tarn lake in foreground, traditional red and white fishing huts scattered along coastline, winding roads and bridges connecting islands, moody overcast sky with warm light on horizon, spectacular Nordic Arctic archipelago landscape",
                "1, 2, 3", 28, 20, 3.5, 3.5, 42,
            ],
        ],
        inputs=[
            model_name, image_input,
            src_prompt, tar_prompt, tar_prompt_neg,
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
