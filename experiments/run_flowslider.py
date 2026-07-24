import os
import sys
import torch
import pandas as pd
from PIL import Image

# ---------------------------------------------------------------------------
# 1. Dynamic Path Resolution (Works regardless of working directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # .../FlowSlider/experiments
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) # .../FlowSlider

# Always route heavy HuggingFace/Torch caches to high-capacity local scratch disk
os.environ["HF_HOME"] = "/tmp/eoikonom/hf_cache"
os.environ["TORCH_HOME"] = "/tmp/eoikonom/torch_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

# Add repo root to sys.path so 'import app' works from anywhere
sys.path.append(REPO_ROOT)

# ---------------------------------------------------------------------------
# 2. HuggingFace Authentication
# ---------------------------------------------------------------------------
from huggingface_hub import login

token_path = os.path.expanduser("~/.hf_token")
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        login(token=f.read().strip())
    print("Logged in via cluster token file.")
elif "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("Logged in via HF_TOKEN environment variable.")
else:
    print("⚠️ Warning: No HF token found! Gated models (like FLUX.1-dev) may fail to download.")

# ---------------------------------------------------------------------------
# 3. Import Application Logic
# ---------------------------------------------------------------------------
try:
    from app import run_edit
except ImportError as e:
    print(f"❌ Failed to import run_edit from app.py: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 4. Configure File Paths Relative to REPO_ROOT
# ---------------------------------------------------------------------------
CSV_PATH = os.path.join(REPO_ROOT, "experiments24-7.csv")
SOURCE_IMG_DIR = os.path.join(REPO_ROOT, "datasets/test_images")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs/flowslider/1d")
STEERING_STEPS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("==================================================")
print(" CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(" Device:", torch.cuda.get_device_name(0))
    print(f" VRAM Free: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f" Target CSV: {CSV_PATH}")
print("==================================================")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Cannot find CSV at '{CSV_PATH}'")

df_experiments = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df_experiments)} test cases.")

# ---------------------------------------------------------------------------
# 5. Benchmark Execution Loop
# ---------------------------------------------------------------------------
for idx, row in df_experiments.iterrows():
    sweep_id = str(row['id'])
    base_prompt = str(row['base_prompt'])
    subprompt1 = str(row['subprompt_1'])
    seed = int(row['seed'])

    exp_dir = os.path.join(OUTPUT_DIR, sweep_id)
    os.makedirs(exp_dir, exist_ok=True)

    source_img_path = os.path.join(SOURCE_IMG_DIR, f"{sweep_id}.png")
    if not os.path.exists(source_img_path):
        print(f"⚠️ [{idx+1}/{len(df_experiments)}] Skipping '{sweep_id}' (Missing image at {source_img_path})")
        continue

    scales_str = ", ".join(map(str, STEERING_STEPS))
    clean_prompt = "".join(c for c in subprompt1 if c.isalnum() or c in (" ", "_")).replace(" ", "_")

    print(f"\n[{idx+1}/{len(df_experiments)}] Processing '{sweep_id}' locally on GPU...")

    try:
        input_image = Image.open(source_img_path).convert("RGB")

        results = run_edit(
            model_name="FLUX.1-dev",
            image=input_image,
            src_prompt=base_prompt,
            tar_prompt=subprompt1,
            tar_prompt_neg="",
            strengths_str=scales_str,
            T_steps=28,
            n_max=20,
            src_cfg=3.5,
            tar_cfg=3.5,
            seed=float(seed)
        )

        steered_images = []
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                img_obj, label = item
                if label != "Original":
                    steered_images.append(img_obj)
            elif isinstance(item, Image.Image):
                steered_images.append(item)

        # Save each steered image matching STEERING_STEPS
        for i, out_img in enumerate(steered_images):
            s_val = STEERING_STEPS[i] if i < len(STEERING_STEPS) else i
            s_str = f"{s_val}".replace(".", "_")
            save_path = os.path.join(exp_dir, f"{sweep_id}_s_{s_str}_{clean_prompt}.png")
            
            # Ensure image is in RGB format before saving as PNG
            if out_img.mode != "RGB":
                out_img = out_img.convert("RGB")
                
            out_img.save(save_path)
            print(f"  -> Saved: {save_path}")

    except Exception as e:
        print(f"❌ Execution error for {sweep_id}: {e}")

print("\n🎉 FlowSlider Benchmark Complete!")
