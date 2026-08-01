import os
import sys
import json
import glob
import torch
import argparse
import pandas as pd
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Mock Application with Unique Alpha Step Simulation & Validation
# ---------------------------------------------------------------------------
class MockAppEdit:
    def __call__(self, model_name, image, src_prompt, tar_prompt, tar_prompt_neg, strengths_str, T_steps, n_max, src_cfg, tar_cfg, seed):
        from PIL import ImageEnhance
        
        strengths = [float(s.strip()) for s in strengths_str.split(",")]
        results = []
        generated_variants = []

        print(f"\n[Mock Pipeline] Running alpha sweep for prompt: '{tar_prompt}'")

        for s in strengths:
            # Simulate a visible, unique difference per alpha step
            img_np = np.array(image).astype(np.float32)
            factor = 1.0 + (s * 0.2)
            modified_np = np.clip(img_np * factor, 0, 255).astype(np.uint8)
            dummy_img = Image.fromarray(modified_np)

            generated_variants.append(modified_np)
            results.append((dummy_img, f"Step_{s}"))
            print(f"  -> Alpha {s:4.1f} | Pixel mean value: {np.mean(modified_np):.2f}")

        # Automatically run uniqueness assertions during mock execution
        for i in range(len(generated_variants) - 1):
            assert not np.array_equal(generated_variants[i], generated_variants[i+1]), \
                f"❌ Failure: Alpha steps {strengths[i]} and {strengths[i+1]} produced identical images!"

        print("✅ Success: All alpha steps produced uniquely modified outputs during this sweep!\n")
        return results

def run(dataset_type, mapping_file, images_dir, output_dir, hf_token_path, start_idx, end_idx, lora_path, dry_run):
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
    if not dry_run:
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
            print("⚠️ Warning: No HF token found! Gated models may fail to download.")

    # ---------------------------------------------------------------------------
    # 3. Import Application Logic or Mock
    # ---------------------------------------------------------------------------
    if dry_run:
        print("[DRY RUN MODE] Using MockAppEdit to bypass model loading.")
        run_edit = MockAppEdit()
    else:
        try:
            from app import run_edit
        except ImportError as e:
            print(f"❌ Failed to import run_edit from app.py: {e}")
            sys.exit(1)

    steering_steps = [0.0, 0.5, 1.0, 1.5, 2.0]
    os.makedirs(output_dir, exist_ok=True)

    print("==================================================")
    print(" Dry Run Mode:", dry_run)
    print(" CUDA Available:", torch.cuda.is_available())
    print(f" Target Mapping Source: {mapping_file}")
    print("==================================================")

    # ---------------------------------------------------------------------------
    # 4. Load Dataset (Parquet for PIE-bench or JSON for rs-objects)
    # ---------------------------------------------------------------------------
    dataset_records = []

    if dataset_type == "pie-bench":
        print(f"Loading PIE-bench parquet files from {mapping_file}...")
        if os.path.isdir(mapping_file):
            parquet_files = sorted(glob.glob(os.path.join(mapping_file, "**/*.parquet"), recursive=True))
        else:
            parquet_files = [mapping_file]
            
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for PIE-bench at {mapping_file}")
            
        for p_file in parquet_files:
            df = pd.read_parquet(p_file)
            for idx, row in df.iterrows():
                sample_id = str(row.get("id", f"sample_{idx}"))
                target_prompt = str(row.get("target_prompt", ""))
                source_prompt = str(row.get("source_prompt", ""))
                
                img_obj = row.get("image", None)
                img_path = None
                
                if isinstance(img_obj, dict) and "bytes" in img_obj:
                    temp_img_dir = os.path.join(images_dir, "_extracted_cache")
                    os.makedirs(temp_img_dir, exist_ok=True)
                    img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                    if not os.path.exists(img_path):
                        with open(img_path, "wb") as f_img:
                            f_img.write(img_obj["bytes"])
                elif isinstance(img_obj, bytes):
                    temp_img_dir = os.path.join(images_dir, "_extracted_cache")
                    os.makedirs(temp_img_dir, exist_ok=True)
                    img_path = os.path.join(temp_img_dir, f"{sample_id}.jpg")
                    if not os.path.exists(img_path):
                        with open(img_path, "wb") as f_img:
                            f_img.write(img_obj)
                else:
                    img_filename = str(row.get("path", f"{sample_id}.jpg"))
                    img_path = os.path.join(images_dir, img_filename)

                dataset_records.append({
                    "id": sample_id,
                    "source_prompt": source_prompt,
                    "target_prompt": target_prompt,
                    "image_path": img_path,
                    "seed": 42
                })
    else:
        if not os.path.exists(mapping_file):
            raise FileNotFoundError(f"Cannot find mapping file at '{mapping_file}'")
            
        print(f"Loading JSON mapping from {mapping_file}...")
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)

        if isinstance(mapping_data, dict):
            mapping_data = [{"id": k, **v} for k, v in mapping_data.items()]

        for item in mapping_data:
            sweep_id = str(item.get('id', 'sample'))
            base_prompt = str(item.get('base_prompt', item.get('source_prompt', '')))
            subprompt1 = str(item.get('target_prompt', item.get('subprompt_1', '')))
            seed = int(item.get('seed', 42))
            img_filename = item.get('image', item.get('file_name', f"{sweep_id}.png"))
            source_img_path = os.path.join(images_dir, img_filename)

            dataset_records.append({
                "id": sweep_id,
                "source_prompt": base_prompt,
                "target_prompt": subprompt1,
                "image_path": source_img_path,
                "seed": seed
            })

    actual_end_idx = end_idx if end_idx is not None else len(dataset_records)
    dataset_slice = dataset_records[start_idx:actual_end_idx]
    print(f"Processing slice [{start_idx}:{actual_end_idx}] out of {len(dataset_records)} total items.")

    # ---------------------------------------------------------------------------
    # 5. Benchmark Execution Loop
    # ---------------------------------------------------------------------------
    for idx, row in enumerate(dataset_slice):
        current_idx = start_idx + idx
        sweep_id = row['id']
        base_prompt = row['source_prompt']
        subprompt1 = row['target_prompt']
        source_img_path = row['image_path']
        seed = row['seed']

        # Debug prompt checks placed correctly after variables are initialized
        print(f"\n--- DEBUG PROMPT CHECK ---")
        print(f"Sample ID: {sweep_id}")
        print(f"Source Prompt: {base_prompt}")
        print(f"Target Prompt: {subprompt1}")
        print(f"--------------------------")

        exp_dir = os.path.join(output_dir, str(sweep_id))
        os.makedirs(exp_dir, exist_ok=True)

        if not os.path.exists(source_img_path):
            if dry_run:
                os.makedirs(os.path.dirname(source_img_path), exist_ok=True)
                Image.new("RGB", (256, 256), color="gray").save(source_img_path)
            else:
                print(f"⚠️ [{current_idx+1}/{actual_end_idx}] Skipping '{sweep_id}' (Missing image at {source_img_path})")
                continue

        scales_str = ", ".join(map(str, steering_steps))
        print(f"\n[{current_idx+1}/{actual_end_idx}] Processing '{sweep_id}'...")

        try:
            input_image = Image.open(source_img_path).convert("RGB")
            input_image.save(os.path.join(exp_dir, "source_original.png"), format="PNG")

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
                step_idx_str = f"{i:02d}"
                save_path = os.path.join(exp_dir, f"step_{step_idx_str}.png")
                
                if out_img.mode != "RGB":
                    out_img = out_img.convert("RGB")
                    
                out_img.save(save_path)
                print(f"  -> Saved step {step_idx_str}: {save_path}")

        except Exception as e:
            print(f"❌ Execution error for {sweep_id}: {e}")

    print("\n🎉 FlowSlider Benchmark Execution Complete!")


def main():
    parser = argparse.ArgumentParser(description="Run FlowSlider Benchmarks")
    parser.add_argument("--dataset_type", type=str, required=True, choices=["pie-bench", "anytext", "rs-objects"])
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--hf_token_path", type=str, default="~/.hf_token")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Run with mock pipeline to debug script logic without VRAM/models")
    
    args = parser.parse_args()

    run(
        dataset_type=args.dataset_type,
        mapping_file=args.mapping_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        hf_token_path=args.hf_token_path,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        lora_path=args.lora_path,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()