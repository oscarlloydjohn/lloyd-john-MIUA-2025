import os
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse

# Deletes all files that contain "_Scaled_2_" in their name
def remove_files_containing(data_path, string):
    
    # Get all files in the directory
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    
    # Filter out files containing "_Scaled_2_"
    files_to_remove = [f for f in files if string in f]
    
    # Define a helper function to remove the file
    def delete_file(file_name):
        
        try:
            
            file_path = os.path.join(data_path, file_name)
            
            os.remove(file_path)  # or shutil.rmtree(file_path) if it is a directory
            
            print(f"Deleted: {file_name}")
            
        except Exception as e:
            
            print(f"Error deleting {file_name}: {e}")

    # Use ThreadPoolExecutor to delete files in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(delete_file, files_to_remove)


# Use freesurfer mri_convert in parallel
def batch_mgz_to_nii(data_path, batchname):
    
    dirname = os.path.splitext(os.path.basename(batchname))[0]
    
    target = os.path.join(data_path, dirname, "mri")
    
    # List all .mgz files in the given directory (non-recursive)
    mgz_files = [f for f in os.listdir(target) if f.endswith('.mgz')]
    
    print(mgz_files)
    
    # Convert individual file
    def convert_mgz_to_nii(filename):
        file_path = os.path.join(target, filename)

        # Define output filename with .nii extension
        output_filename = os.path.splitext(filename)[0] + ".nii"
        output_file_path = os.path.join(target, output_filename)

        process = subprocess.Popen(["mri_convert", file_path, output_file_path])
        process.wait() 

        print(f"Converted {filename} to {output_filename}")
    
    # Use ThreadPoolExecutor to run the tasks in parallel
    with ThreadPoolExecutor() as executor:
        
        # Submit all conversion tasks to the executor
        executor.map(convert_mgz_to_nii, mgz_files)
    
    print("All .mgz files have been converted.")
    
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

# Gets a list of nii files in data_path. Non recursive
def list_nii(data_path):
    
    nii_files = [f for f in os.listdir(data_path) if f.endswith('.nii')]
        
    return nii_files

# Runs fastsurfer seg only on a given nii file, putting output in a directory of the same name
def process_file(data_path, filename, license_path, container_path, threads):
    
    dirname = os.path.splitext(os.path.basename(filename))[0]

    command = [
        "/fastsurfer/run_fastsurfer.sh",
        "--fs_license", f"{license_path}",
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
        # Define the function to run mri_convert for each file
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
        
        with open(f"{data_path}/completed.log", 'r') as completed_file:
            
            completed_files = set(line.strip() for line in completed_file)
            
    except FileNotFoundError:
        
        print("completed.txt not found, starting fresh.")
    
    for file in nii_list:
        
        # Skip the file if it's already in the completed.txt file
        
        if file in completed_files:
            
            print(f"Skipping {file}, already processed.")
            
            continue

        try:
            
            process_file(data_path, file, license_path, container_path, 12)
            
            move_nii(data_path, file)
            
            batch_mgz_to_nii(data_path, file)

            # Open 'completed.txt' file in append mode and write the processed file
            with open(f"{data_path}/completed.log", 'a') as completed_file:
                
                completed_file.write(f"{file}\n")

        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

    print("Batch processing complete. Files have been logged to completed.log.")

parser = argparse.ArgumentParser(description="Script for batch processing nii files using fastsurfer")

parser.add_argument('--compact_dir', action='store_true')
parser.add_argument('--remove_files_containing', type=str, required=False)
parser.add_argument('--data_path', type=str, required=True)
parser.add_argument('--license_path', type=str, required=True)
parser.add_argument('--container_path', type=str, required=True)

args = parser.parse_args()

if args.compact_dir:
    compact_dir(args.data_path)

if args.remove_files_containing:
    remove_files_containing(args.data_path, args.remove_files_containing)

# Perform batch run with other parameters
batch_run(args.data_path, list_nii(args.data_path), args.container_path, args.license_path)
