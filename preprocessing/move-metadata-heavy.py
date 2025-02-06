import os
import shutil

def find_files_recursively(directory):
    """Recursively find all files in the directory."""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def move_matching_files(source_dir, target_dir):
    # Get a list of all files in the source directory (top level only)
    source_files = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    
    # Get a list of all files in the target directory (recursively)
    target_files = find_files_recursively(target_dir)
    
    # Create a dictionary of target files with the last 14 characters of the base name as keys
    target_dict = {os.path.splitext(os.path.basename(f))[-14:]: f for f in target_files}
    
    # List and move files from source to target if the last 14 characters of the base name match
    for file in source_files:
        base_name = os.path.splitext(os.path.basename(file))[-14:]
        if base_name in target_dict:
            src_file = file
            dest_file = target_dict[base_name]
            print(f"Match found: {src_file} -> {dest_file}")
            shutil.move(src_file, dest_file)
            print(f"Moved {src_file} to {dest_file}")

# Paths to the source and target directories
source_dir = "/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/adni1-complete-1yr-3t/metadata-heavy"
target_dir = "/vol/scratch/SoC/misc/2024/sc22olj/full-datasets/adni1-complete-1yr-3t/data"

move_matching_files(source_dir, target_dir)
