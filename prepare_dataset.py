import os
import argparse
import random
import io
from PIL import Image
import shutil
from pathlib import Path

def get_image_files(directory):
    """Retrieve all image files from a given directory."""
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if Path(filename).suffix.lower() in valid_exts:
                files.append(os.path.join(root, filename))
    return files

def get_parquet_files(directory):
    """Retrieve all parquet files from a given directory or return the file if it's already a parquet file."""
    if os.path.isfile(directory) and directory.endswith('.parquet'):
        return [directory]
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith('.parquet'):
                files.append(os.path.join(root, filename))
    return files

def extract_parquet_images(parquet_paths):
    """Reads image bytes from parquet files and saves them to a 'raw' directory next to each file."""
    import pandas as pd
    from tqdm import tqdm
    extracted_files = []
    
    print(f"Extracting images from {len(parquet_paths)} parquet files...")
    for p_path in parquet_paths:
        parent_dir = os.path.dirname(os.path.abspath(p_path))
        raw_dir = os.path.join(parent_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        try:
            df = pd.read_parquet(p_path)
            print(f"[{os.path.basename(p_path)}] Columns found in Parquet: {df.columns.tolist()}")
            img_col = None
            if 'image' in df.columns:
                img_col = 'image'
            elif 'image.bytes' in df.columns:
                img_col = 'image.bytes'

            if img_col:
                print(f"Processing {os.path.basename(p_path)} ({len(df)} rows) using column '{img_col}'")
                # Use tqdm to show progress bar
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {os.path.basename(p_path)}"):
                    img_data = row[img_col]
                    
                    img_bytes = None
                    ext = '.jpg'
                    
                    if isinstance(img_data, dict) and 'bytes' in img_data:
                        img_bytes = img_data['bytes']
                        if 'path' in img_data and img_data['path']:
                            ext = Path(img_data['path']).suffix or '.jpg'
                    elif isinstance(img_data, bytes):
                        img_bytes = img_data
                        
                    if img_bytes is not None:
                        filename = f"{Path(p_path).stem}_{idx}{ext}"
                        out_path = os.path.join(raw_dir, filename)
                        
                        try:
                            # Skip if already exists to save time on reruns
                            if not os.path.exists(out_path):
                                image = Image.open(io.BytesIO(img_bytes))
                                image.save(out_path)
                            extracted_files.append(out_path)
                        except Exception as e:
                            pass # Skip rows that aren't valid images silently based on progress bar
            else:
                print(f"Warning: Neither 'image' nor 'image.bytes' column found in {p_path}.")
        except Exception as e:
            print(f"Failed to read Parquet file {p_path}: {e}")
            
    return extracted_files

def create_symlinks(file_paths, target_dir):
    """Create symbolic links for a list of file paths in the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    for file_path in file_paths:
        src = os.path.abspath(file_path)
        dst = os.path.join(target_dir, os.path.basename(file_path))
        
        # In case there's a filename collision, append a random string
        while os.path.exists(dst):
            name, ext = os.path.splitext(os.path.basename(file_path))
            dst = os.path.join(target_dir, f"{name}_{random.randint(1000, 9999)}{ext}")
            
        os.symlink(src, dst)

def load_domain_files(path):
    """Determines if a path holds parquet or image files, extracts them if parquet, and returns a list of paths."""
    parquet_files = get_parquet_files(path)
    if parquet_files:
        print(f"Detected Parquet files for {path}. Extracting to source directory 'raw' folder...")
        return extract_parquet_images(parquet_files)
    else:
        return get_image_files(path)

def main():
    parser = argparse.ArgumentParser(description="Format dataset into CycleGAN structure using symlinks.")
    parser.add_argument('--dir_A', type=str, required=True, help="Path to the original Domain A dataset")
    parser.add_argument('--dir_B', type=str, required=True, help="Path to the original Domain B dataset")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the new dataset to create")
    parser.add_argument('--output_root', type=str, default='./datasets', help="Root directory for the output dataset")
    parser.add_argument('--val_ratio', type=float, default=0.1, help="Fraction of images to reserve for validation (default: 0.1)")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="Fraction of images to reserve for testing (default: 0.1)")
    parser.add_argument('--max_images', type=int, default=2000, help="Maximum number of images to sample per domain (default: 2000)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for train/test split")
    parser.add_argument('--method', type=str, choices=['symlink', 'copy'], default='symlink', help='Method to link files: "symlink" (default, with auto-fallback to copy) or "copy" (safer for Docker across drives)')
    
    args = parser.parse_args()
    
    # 1. Gather all files
    files_A = load_domain_files(args.dir_A)
    files_B = load_domain_files(args.dir_B)
    
    print(f"Found {len(files_A)} images in {args.dir_A}")
    print(f"Found {len(files_B)} images in {args.dir_B}")
    
    if len(files_A) == 0 or len(files_B) == 0:
        print("Error: One or both source directories contain no images. Exiting.")
        return

    # 2. Shuffle and balance dataset sizes based on max_images
    random.seed(args.seed)
    random.shuffle(files_A)
    random.shuffle(files_B)
    
    min_length = min(len(files_A), len(files_B), args.max_images)
    print(f"Equalizing domain sizes and capping to {args.max_images}... Target files per domain: {min_length}")
    
    files_A = files_A[:min_length]
    files_B = files_B[:min_length]
    
    # 3. Train/Val/Test Split (e.g., 8:1:1)
    num_val = int(min_length * args.val_ratio)
    num_test = int(min_length * args.test_ratio)
    
    # Slices for A
    test_A = files_A[:num_test]
    val_A = files_A[num_test:num_test + num_val]
    train_A = files_A[num_test + num_val:]
    
    # Slices for B
    test_B = files_B[:num_test]
    val_B = files_B[num_test:num_test + num_val]
    train_B = files_B[num_test + num_val:]
    
    print(f"Split completed. Train: {len(train_A)}, Val: {len(val_A)}, Test: {len(test_A)}")
    
    # 4. Create output directory structure
    dataset_out_dir = os.path.join(args.output_root, args.dataset_name)
    print(f"Creating CycleGAN dataset structure at: {dataset_out_dir}")
    
    # 5. Create Symlinks
    dirs_to_link = [
        ("trainA", train_A), ("valA", val_A), ("testA", test_A),
        ("trainB", train_B), ("valB", val_B), ("testB", test_B)
    ]
    
    for folder_name, file_list in dirs_to_link:
        print(f"Linking {len(file_list)} files -> {folder_name}")
        create_symlinks(file_list, os.path.join(dataset_out_dir, folder_name))
        
    print("Done! Dataset is ready.")

if __name__ == "__main__":
    main()
