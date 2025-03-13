import os
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse


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

        output_filename = os.path.splitext(filename)[0] + ".nii"
        
        output_file_path = os.path.join(target, output_filename)

        process = subprocess.Popen(["mri_convert", file_path, output_file_path])
        
        process.wait() 

        print(f"Converted {filename} to {output_filename}")
    
    with ThreadPoolExecutor() as executor:
        
        executor.map(convert_mgz_to_nii, mgz_files)
        
    print("All .mgz files have been converted.")
    
    return

# Gets a list of nii files in data_path. Non recursive
def list_nii(data_path):
    
    nii_files = [f for f in os.listdir(data_path) if f.endswith('.nii')]
        
    return nii_files

# Runs fastsurfer seg only on a given nii file, putting output in a directory of the same name
def process_file(data_path, filename, license_path, threads, tesla3=False):
    
    dirname = os.path.splitext(os.path.basename(filename))[0]

    command = [
        "/fastsurfer/run_fastsurfer.sh",
        "--fs_license", f"{license_path}",
        f"--sd", data_path,
        "--sid", dirname,
        f"--t1", f"{data_path}/{filename}",
        "--seg_only", "--threads", f"{threads}"
    ]
    
    if tesla3:
        
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

# Move an nii file into the directory under the same name
def move_nii(data_path, filename):
    
    dirname = os.path.splitext(os.path.basename(filename))[0]

    target = os.path.join(data_path, dirname)

    if not os.path.exists(target):
        
        print(f"Directory {target} does not exist")
        
        return
    
    target_path = os.path.join(target, filename)

    try:
        
        shutil.move(os.path.join(data_path, filename), target_path)
        
        print(f"Moved {filename} to {target_path}")
        
    except Exception as e:
        
        print(f"Error moving file {filename}: {e}")
        
    return

def batch_run(data_path, nii_list, args):
    
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
            
            process_file(data_path, file, args.license_path, args.threads, args.tesla3)
            
            if args.keep_orig:
            
                move_nii(data_path, file)
                
            else:
                
                os.remove(os.path.join(data_path, file))
            
            if args.nii_to_mgz:
                
                batch_mgz_to_nii(data_path, file)

            # Open 'completed.txt' file in append mode and write the processed file
            with open(f"{data_path}/completed.log", 'a') as completed_file:
                
                completed_file.write(f"{file}\n")

        except Exception as e:
            
            print(f"Error processing {file}: {e}")
            
            continue

    print("Batch processing complete. Files have been logged to completed.log.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script for batch processing nii files using fastsurfer")

    parser.add_argument('--keep_orig', action='store_true')
    parser.add_argument('--mgz_to_nii', action='store_true')
    parser.add_argument('--tesla3', action='store_true')
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--license_path', type=str, required=True)
    parser.add_argument('--container_path', type=str, required=True)

    args = parser.parse_args()

    batch_run(args.data_path, list_nii(args.data_path), args)
