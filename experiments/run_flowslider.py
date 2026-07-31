import os
import sys
import json
import torch
import argparse
import pandas as pd
from PIL import Image

def run(dataset_type, mapping_file, images_dir, output_dir, hf_token_path, start_idx, end_idx, lora_path):
    # ---------------------------------------------------------------------------
    # 1. Dynamic Path Resolution & Cache Routing
    # ---------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))

    os.environ["HF_HOME"] = "/tmp/eoikonom/hf_cache"
    os.environ["TORCH_HOME"] = "/tmp/eoikonom/torch_cache"
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

    sys.path.append(repo_root)

    # ---------------------------------------------------------------------------
    # 2. HuggingFace Authentication
    # ---------------------------------------------------------------------------
    from huggingface_hub import login

    token_path = os.path.expanduser(hf_token_path)
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

    steering_steps = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================")
    print(" CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(" Device:", torch.cuda.get_device_name(0))
        print(f" VRAM Free: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f" Target Mapping File: {mapping_file}")
    print("==================================================")

    if lora_path:
        print(f"🔧 LoRA path provided: {lora_path}")

    # ---------------------------------------------------------------------------
    # 4. Load Dataset Mapping File (JSON)
    # ---------------------------------------------------------------------------
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Cannot find mapping file at '{mapping_file}'")

    print(f"Loading dataset mapping from {mapping_file}...")
    with open(mapping_file, 'r') as f:
        dataset_records = json.load(f)

    if isinstance(dataset_records, dict):
        dataset_records = [{"id": k, **v} for k, v in dataset_records.items()]

    actual_end_idx = end_idx if end_idx is not None else len(dataset_records)
    dataset_slice = dataset_records[start_idx:actual_end_idx]
    print(f"Processing slice [{start_idx}:{actual_end_idx}] out of {len(dataset_records)} total items.")

    # ---------------------------------------------------------------------------
    # 5. Benchmark Execution Loop
    # ---------------------------------------------------------------------------
    for idx, row in enumerate(dataset_slice):
        current_idx = start_idx + idx
        sweep_id = str(row.get('id', current_idx))
        base_prompt = str(row.get('base_prompt', row.get('source_prompt', '')))
        subprompt1 = str(row.get('target_prompt', row.get('subprompt_1', '')))
        seed = int(row.get('seed', 42))

        img_filename = row.get('image', row.get('file_name', f"{sweep_id}.png"))
        source_img_path = os.path.join(images_dir, img_filename)

        exp_dir = os.path.join(output_dir, str(sweep_id))
        os.makedirs(exp_dir, exist_ok=True)

        if not os.path.exists(source_img_path):
            print(f"⚠️ [{current_idx+1}/{actual_end_idx}] Skipping '{sweep_id}' (Missing image at {source_img_path})")
            continue

        scales_str = ", ".join(map(str, steering_steps))
        clean_prompt = "".join(c for c in subprompt1 if c.isalnum() or c in (" ", "_")).replace(" ", "_")

        print(f"\n[{current_idx+1}/{actual_end_idx}] Processing '{sweep_id}' locally on GPU...")

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

            for i, out_img in enumerate(steered_images):
                s_val = steering_steps[i] if i < len(steering_steps) else i
                s_str = f"{s_val}".replace(".", "_")
                save_path = os.path.join(exp_dir, f"{sweep_id}_s_{s_str}_{clean_prompt}.png")
                
                if out_img.mode != "RGB":
                    out_img = out_img.convert("RGB")
                    
                out_img.save(save_path)
                print(f"  -> Saved: {save_path}")

        except Exception as e:
            print(f"❌ Execution error for {sweep_id}: {e}")

    print("\n🎉 FlowSlider Benchmark Complete!")


def main():
    parser = argparse.ArgumentParser(description="Run FlowSlider Benchmarks")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["pie-bench", "anytext", "rs-objects"], help="Type of dataset being evaluated")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to the JSON mapping file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to the directory containing dataset images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save benchmark output images")
    parser.add_argument("--hf_token_path", type=str, default="~/.hf_token", help="Path to HuggingFace token file")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for dataset slice")
    parser.add_argument("--end_idx", type=int, default=None, help="Ending index for dataset slice")
    parser.add_argument("--lora_path", type=str, default=None, help="Optional FlowSlider / LoRA weight path")
    
    args = parser.parse_args()

    run(
        dataset_type=args.dataset_type,
        mapping_file=args.mapping_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        hf_token_path=args.hf_token_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        lora_path=args.lora_path
    )


if __name__ == "__main__":
    main()