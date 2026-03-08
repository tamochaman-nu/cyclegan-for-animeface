import os
import argparse
import random
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

def create_links(file_paths, target_dir, method='symlink'):
    """Create links (or copy) for a list of file paths in the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    for file_path in file_paths:
        src = os.path.abspath(file_path)
        dst = os.path.join(target_dir, os.path.basename(file_path))
        
        # In case there's a filename collision, append a random string
        while os.path.exists(dst):
            name, ext = os.path.splitext(os.path.basename(file_path))
            dst = os.path.join(target_dir, f"{name}_{random.randint(1000, 9999)}{ext}")
            
        if method == 'copy':
            shutil.copy2(src, dst)
        else:
            try:
                # Docker Linuxコンテナ内でも解決できるよう、極力相対パスでシンボリックリンクを作成
                rel_src = os.path.relpath(src, start=os.path.dirname(dst))
                os.symlink(rel_src, dst)
            except ValueError:
                # Windowsで別ドライブ間のため相対パスが作れない場合（絶対パスリンクはDocker内で壊れるためコピーにフォールバック）
                shutil.copy2(src, dst)
            except OSError:
                # シンボリックリンク作成権限がないなどのエラー時もコピーにフォールバック
                shutil.copy2(src, dst)

def main():
    parser = argparse.ArgumentParser(description="Format dataset into CycleGAN structure using symlinks.")
    parser.add_argument('--dir_A', type=str, required=True, help="Path to the original Domain A dataset")
    parser.add_argument('--dir_B', type=str, required=True, help="Path to the original Domain B dataset")
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the new dataset to create")
    parser.add_argument('--output_root', type=str, default='./datasets', help="Root directory for the output dataset")
    parser.add_argument('--test_ratio', type=float, default=0.1, help="Fraction of images to reserve for testing (default: 0.1)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for train/test split")
    parser.add_argument('--method', type=str, choices=['symlink', 'copy'], default='symlink', help='Method to link files: "symlink" (default, with auto-fallback to copy) or "copy" (safer for Docker across drives)')
    
    args = parser.parse_args()
    
    # 1. Gather all files
    files_A = get_image_files(args.dir_A)
    files_B = get_image_files(args.dir_B)
    
    print(f"Found {len(files_A)} images in {args.dir_A}")
    print(f"Found {len(files_B)} images in {args.dir_B}")
    
    if len(files_A) == 0 or len(files_B) == 0:
        print("Error: One or both source directories contain no images. Exiting.")
        return

    # 2. Shuffle and balance dataset sizes
    random.seed(args.seed)
    random.shuffle(files_A)
    random.shuffle(files_B)
    
    min_length = min(len(files_A), len(files_B))
    print(f"Equalizing domain sizes to the minimum: {min_length} images per domain")
    
    files_A = files_A[:min_length]
    files_B = files_B[:min_length]
    
    # 3. Train/Test Split
    num_test_A = int(len(files_A) * args.test_ratio)
    num_test_B = int(len(files_B) * args.test_ratio)
    
    test_A = files_A[:num_test_A]
    train_A = files_A[num_test_A:]
    
    test_B = files_B[:num_test_B]
    train_B = files_B[num_test_B:]
    
    # 3. Create output directory structure
    dataset_out_dir = os.path.join(args.output_root, args.dataset_name)
    print(f"Creating CycleGAN dataset structure at: {dataset_out_dir}")
    
    # 4. Create Links
    print(f"Linking/Copying {len(train_A)} files -> trainA (method: {args.method})")
    create_links(train_A, os.path.join(dataset_out_dir, "trainA"), method=args.method)
    
    print(f"Linking/Copying {len(test_A)} files -> testA")
    create_links(test_A, os.path.join(dataset_out_dir, "testA"), method=args.method)
    
    print(f"Linking/Copying {len(train_B)} files -> trainB")
    create_links(train_B, os.path.join(dataset_out_dir, "trainB"), method=args.method)
    
    print(f"Linking/Copying {len(test_B)} files -> testB")
    create_links(test_B, os.path.join(dataset_out_dir, "testB"), method=args.method)
    
    print("Done! Dataset is ready.")

if __name__ == "__main__":
    main()
