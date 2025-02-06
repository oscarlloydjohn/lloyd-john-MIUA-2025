import os
import shutil

def check_structure(dir1, dir2):
    for root, dirs, files in os.walk(dir1):
        relative_path = os.path.relpath(root, dir1)
        corresponding_dir2 = os.path.join(dir2, relative_path)
        
        if not os.path.exists(corresponding_dir2):
            print(f"Structure mismatch: {corresponding_dir2} does not exist in {dir2}")
            return False
        
        dirs2 = [d for d in os.listdir(corresponding_dir2) if os.path.isdir(os.path.join(corresponding_dir2, d))]
        if set(dirs) != set(dirs2):
            print(f"Structure mismatch in {relative_path}: {dirs} != {dirs2}")
            return False
    
    return True

def combine_directories(dir1, dir2):
    for root, dirs, files in os.walk(dir2):
        relative_path = os.path.relpath(root, dir2)
        target_dir = os.path.join(dir1, relative_path)
        
        os.makedirs(target_dir, exist_ok=True)
        
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(target_dir, file)
            shutil.copy2(src_file, dest_file)
            print(f"Copied {src_file} to {dest_file}")

# Paths to the two directories
dir1 = "/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/adni1-complete-1yr-3t/data"
dir2 = "/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/adni1-complete-1yr-3t/metadata-light"

if check_structure(dir1, dir2):
    combine_directories(dir1, dir2)
else:
    print("The directory structures do not match.")