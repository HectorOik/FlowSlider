import os
import sys
import json
import argparse

def debug_dataset_loading(mapping_file, images_dir, start_idx, end_idx):
    print("=" * 60)
    print(" DEBUGGING: Dataset Mapping & Path Resolution")
    print("=" * 60)

    # 1. Check mapping file existence
    if not os.path.exists(mapping_file):
        print(f"❌ ERROR: Mapping file not found at: {mapping_file}")
        return False
    print(f"✅ Mapping file found: {mapping_file}")

    # 2. Check images directory existence
    if not os.path.exists(images_dir):
        print(f"❌ ERROR: Images directory not found at: {images_dir}")
        return False
    print(f"✅ Images directory found: {images_dir}")

    # 3. Load and parse JSON
    try:
        with open(mapping_file, 'r') as f:
            dataset_records = json.load(f)
        print(f"✅ Successfully parsed JSON. Total records: {len(dataset_records)}")
    except Exception as e:
        print(f"❌ ERROR: Failed to parse JSON mapping file: {e}")
        return False

    # Normalize structure for PIE-Bench dictionary format (where keys are image filenames)
    if isinstance(dataset_records, dict):
        dataset_records = [{"id": k, **v} for k, v in dataset_records.items()]

    # Apply slicing
    actual_end_idx = end_idx if end_idx is not None else len(dataset_records)
    dataset_slice = dataset_records[start_idx:actual_end_idx]
    print(f" Inspecting slice range: [{start_idx} : {actual_end_idx}] ({len(dataset_slice)} items)")

    # 4. Iterate and validate sample paths & fields
    missing_images_count = 0
    valid_count = 0

    for idx, row in enumerate(dataset_slice):
        current_idx = start_idx + idx
        
        # The 'id' field is the image filename (e.g., '000000000001.jpg')
        sweep_id = str(row.get('id', ''))
        img_filename = sweep_id if sweep_id else f"{current_idx:012d}.jpg"
        
        base_prompt = str(row.get('original_prompt', row.get('base_prompt', row.get('editing_instruction', ''))))
        subprompt1 = str(row.get('editing_prompt', row.get('target_prompt', row.get('editing_instruction', ''))))
        
        source_img_path = os.path.join(images_dir, img_filename)

        # Print first 3 records as a sanity check
        if idx < 3:
            print(f"\n   --- Sample Record #{current_idx} ---")
            print(f"   ID: {sweep_id}")
            print(f"   Base Prompt: {base_prompt}")
            print(f"   Target Prompt: {subprompt1}")
            print(f"   Image Path: {source_img_path}")

        if not os.path.exists(source_img_path):
            if missing_images_count < 5:  # Print first few missing ones
                print(f"   ⚠️ Missing image for item {sweep_id} at: {source_img_path}")
            missing_images_count += 1
        else:
            valid_count += 1

    print("\n" + "=" * 60)
    print(f" DEBUG SUMMARY:")
    print(f"   - Valid images found: {valid_count}/{len(dataset_slice)}")
    print(f"   - Missing images: {missing_images_count}/{len(dataset_slice)}")
    print("=" * 60)
    return True


def debug_app_import():
    print("\n" + "=" * 60)
    print(" DEBUGGING: App Logic & Environment Imports")
    print("=" * 60)

    sys.path.append(os.path.abspath("."))
    
    # Check torch / CUDA
    import torch
    print(f"   - PyTorch Version: {torch.__version__}")
    print(f"   - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU Device: {torch.cuda.get_device_name(0)}")

    # Check app.py import
    try:
        from app import run_edit
        print(f"✅ Successfully imported `run_edit` from `app.py`")
    except ImportError as e:
        print(f"❌ ERROR: Failed to import `run_edit`: {e}")
        return False

    print("=" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Debug FlowSlider Pipeline Logic")
    parser.add_argument("--mapping_file", type=str, required=True, help="Path to JSON mapping file")
    parser.add_argument("--images_dir", type=str, required=True, help="Path to images directory")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=10, help="End index for quick testing (default: 10)")
    
    args = parser.parse_args()

    debug_app_import()
    debug_dataset_loading(
        mapping_file=args.mapping_file,
        images_dir=args.images_dir,
        start_idx=args.start_idx,
        end_idx=args.end_idx
    )

if __name__ == "__main__":
    main()
