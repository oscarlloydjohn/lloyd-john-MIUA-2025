import os
import subprocess
import shutil

# Use freesurfer mri_convert command
def mgz_to_nii(data_path, filename):
    
    dirname = os.path.splitext(os.path.basename(filename))[0]
    
    command = ["mri_convert"]
    
    return

# Recursively search all subdirectories for files and put them in data_path. Delete empty directories
def compact_dir(data_path):
    
    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(data_path, topdown=False):
        
        for filename in files:
            
            file_path = os.path.join(root, filename)
            
            # Move the file to the top-level directory
            shutil.move(file_path, os.path.join(data_path, filename))
        
        # After moving files, check and remove empty directories
        for dirname in dirs:
            
            dir_path = os.path.join(root, dirname)
            
            if not os.listdir(dir_path):  # Directory is empty
                
                os.rmdir(dir_path)
    
    print(f"Compaction of directory {data_path} complete.")

# Example usage
data_path = '/path/to/your/data'
compact_dir(data_path)


# Gets a list of nii files in data_path. Non recursive
def list_nii(data_path):
    
    nii_files = [f for f in os.listdir(data_path) if f.endswith('.nii')]
        
    return nii_files

# Runs fastsurfer seg only on a given nii file, putting output in a directory of the same name
def process_file(data_path, filename, license_path, container_path, threads):
    
    dirname = os.path.splitext(os.path.basename(filename))[0]

    command = [
        "singularity", "exec", "--nv",
        "--no-home",
        f"-B {data_path}:/{data_path}",
        f"-B {license_path}:/{license_path}",
        container_path,
        "/fastsurfer/run_fastsurfer.sh",
        "--fs_license", f"{license_path}/license.txt",
        f"--sd", data_path,
        "--sid", dirname,
        f"--t1", f"{data_path}/{filename}",
        "--3T", "--seg_only", "--threads", f"{threads}"
    ]

    process = subprocess.Popen(command)
    process.wait()
    
    return

def move_nii(data_path, filename):
    
    # Filename without extension for dir
    dirname = os.path.splitext(os.path.basename(filename))[0]

    # Check if target directory exists
    target = os.path.join(data_path, dirname)
    
    if not os.path.exists(target):
        
        print(f"Directory {target} does not exist")
        
        return

    # Create the full target path for the file
    target_path = os.path.join(target, filename)

    # Move the file to the target directory
    try:
        
        shutil.move(os.path.join(data_path, filename), target_path)
        
        print(f"Moved {filename} to {target_path}")
        
    except Exception as e:
        
        print(f"Error moving file {filename}: {e}")
        
    return

def batch_run(data_path, nii_list, container_path, license_path):
    
    # Read the completed.txt file to get a list of already processed files
    completed_files = set()
    
    try:
        with open(f"{data_path}/completed.txt", 'r') as completed_file:
            
            completed_files = set(line.strip() for line in completed_file)
            
    except FileNotFoundError:
        
        print("completed.txt not found, starting fresh.")

    # Open the 'completed.txt' file in append mode to record new processed files
    with open(f"{data_path}/completed.txt", 'a') as completed_file:
        
        for file in nii_list:
            
            # Skip the file if it's already in the completed.txt file
            if file in completed_files:
                
                print(f"Skipping {file}, already processed.")
                
                continue

            process_file(data_path, file, license_path, container_path, 12)
            
            move_nii(data_path, file)

            completed_file.write(f"{file}\n")

    print("Batch processing complete. Files have been logged to completed.txt.")
    
license_path = "/vol/scratch/SoC/misc/2024/sc22olj/tools/freesurfer-license"
container_path = "/vol/scratch/SoC/misc/2024/sc22olj/fastsurfer-gpu.sif"
data_path = "/vol/scratch/SoC/misc/2024/sc22olj/test"

batch_run(data_path, list_nii(data_path), container_path, license_path)