import os
import sys
import json
import torch
import pandas as pd
from PIL import Image

# 1. Force HuggingFace to use local scratch space in /tmp to prevent quota errors
os.environ["HF_HOME"] = "/tmp/eoikonom/hf_cache"
os.environ["TORCH_HOME"] = "/tmp/eoikonom/torch_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# 2. Login to HuggingFace
from huggingface_hub import login

token_path = os.path.expanduser("~/.hf_token")
if os.path.exists(token_path):
    with open(token_path, "r") as f:
        login(token=f.read().strip())
    print("Logged in via cluster token file.")
elif "HF_TOKEN" in os.environ:
    login(token=os.environ["HF_TOKEN"])
    print("Logged in via HF_TOKEN environment variable.")

# 3. Add current directory to path and import app logic
sys.path.append(os.path.abspath("."))
try:
    from app import run_edit
except ImportError as e:
    print(f"❌ Failed to import run_edit from app.py: {e}")
    sys.exit(1)

# 4. Config Paths
CSV_PATH = "./experiments24-7.csv"  # Path to your CSV
SOURCE_IMG_DIR = "./datasets/test_images"
OUTPUT_DIR = "./outputs/flowslider/1d"
STEERING_STEPS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check GPU
print("==================================================")
print(" CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(" Device:", torch.cuda.get_device_name(0))
    print(f" VRAM Free: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("==================================================")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Cannot find CSV at {CSV_PATH}")

df_experiments = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df_experiments)} test cases.")

# 5. Main Execution Loop
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
        results = run_edit(
            model_name="FLUX.1-dev",
            image=source_img_path,
            src_prompt=base_prompt,
            tar_prompt=subprompt1,
            tar_prompt_neg="",
            strengths_str=scales_str,
            t_steps=28,
            n_max=20,
            src_cfg=3.5,
            tar_cfg=3.5,
            seed=float(seed)
        )

        gallery_images = results if isinstance(results, list) else [results]
        
        for i, img_entry in enumerate(gallery_images):
            res_path = None
            if isinstance(img_entry, dict):
                img_data = img_entry.get('image', img_entry)
                res_path = img_data.get('path') if isinstance(img_data, dict) else img_data
            elif isinstance(img_entry, str):
                res_path = img_entry

            if res_path and os.path.exists(res_path):
                out_img = Image.open(res_path).convert("RGB")
                s_val = STEERING_STEPS[i] if i < len(STEERING_STEPS) else i
                s_str = f"{s_val}".replace(".", "_")
                save_path = os.path.join(exp_dir, f"{sweep_id}_s_{s_str}_{clean_prompt}.png")
                out_img.save(save_path)
                print(f"  -> Saved: {save_path}")

    except Exception as e:
        print(f"❌ Execution error for {sweep_id}: {e}")

print("\n🎉 Benchmark Complete!")