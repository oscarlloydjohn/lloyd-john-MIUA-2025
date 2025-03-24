import os
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse

"""
Fastsurfer Processing Pipeline Script
======================================

A standalone script that can perform FastSurfer processing on a batch of .nii files.  
The script supports multithreading via FastSurfer’s built-in capabilities.  
It tracks processed files by logging them into a completed.log file within the data_path directory,  
allowing the script to resume without reprocessing completed files. If the file does not exist then it will start from scratch.

See the accompanying run_processing.sh script which creates an apptainer for running this script.


Arguments
---------

- **--data_path** (*str*) **(required)**:  
    The path to the directory containing NIfTI files for processing.

- **--license_path** (*str*) **(required)**:
    The path to the FreeSurfer license file.

- **--threads** (*int*) **(required)**:  
    Number of threads to use for parallel processing.

- **--tesla3** (*flag*):
    Indicate to FastSurfer that the input images are 3T MRI scans, where its default is 1.5T scans.

- **--mgz_to_nii** (*flag*):
    Make a copy of all resulting .mgz files as .nii files in the same directory.

- **--keep_orig** (*flag*):
    Keep the original .nii files after processing. By default, the original files are deleted.


Usage
-----

Run the script inside a container compatible with fastsurfer, with the required arguments:

.. code-block:: bash

    python3 preprocess.py --data_path /path/to/directory --license_path /path/to/license --threads N [--tesla3] [mgz_to_nii] [--keep_orig]

:author: Oscar Lloyd-John
"""


def batch_mgz_to_nii(data_path: os.PathLike[str], batchname: str) -> None:
    """
    Convert all .mgz files in a specified directory to .nii format using mri_convert from freesurfer.

    :param data_path: The path to the dataset directory.
    :type data_path: os.PathLike[str]
    :param batchname: The name of the upper level image directory, i.e the original filename of the nii file excluding the extension. After processing this will simply be a directory rather than a file.
    :type batchname: str
    :returns: None
    
    """
    
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

def list_nii(data_path: os.PathLike[str]) -> list:
    """
    List all .nii files in the directory.

    :param data_path: The path to the dataset directory.
    :type data_path: os.PathLike[str]
    :returns: List of .nii files in the directory.
    :rtype: list

    """

    
    nii_files = [f for f in os.listdir(data_path) if f.endswith('.nii')]
        
    return nii_files

def process_file(data_path: os.PathLike[str], filename: str, license_path: os.PathLike[str], threads: int, tesla3: bool = False):
    """
    Runs fastsurfer seg only on a given nii file, putting output in a directory of the same name. Allows fastsurfer multithreading using the --threads flag.

    :param data_path: The path to the dataset directory.
    :type data_path: os.PathLike[str]
    :param filename: The filename of the nii file to be processed.
    :type filename: str
    :param license_path: The path to the freesurfer license file
    :type license_path: os.PathLike[str]
    :returns: None

    """
    
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

def move_nii(data_path: os.PathLike[str], filename: str) -> None:
    """
    Moves a .nii file to the directory of the same name, intended to be used after preprocessing. Will only be used if the --keep_orig flag is set.

    :param data_path: The path to the dataset directory.
    :type data_path: os.PathLike[str]
    :param filename: The filename of the nii file to be moved.
    :type filename: str
    :returns: None

    """

    
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

def batch_run(data_path: os.PathLike[str] , nii_list: list, args: argparse.Namespace) -> None:
    """
    Runs the fastsurfer pipeline on a batch of nii files, logging the processed files in a completed.log file such that the script can be resumed without reprocessing completed files. Allows original files to be kept or deleted, and .mgz files to be converted to .nii files if desired.

    :param data_path: The path to the dataset directory.
    :type data_path: os.PathLike[str]
    :param nii_list: A list of .nii files to be processed, obtained from list_nii.
    :type nii_list: list
    :param args: The arguments passed to the script.
    :type args: argparse.Namespace
    :returns: None
    
    """
    
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
            
            if args.mgz_to_nii:
                
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

    args = parser.parse_args()

    batch_run(args.data_path, list_nii(args.data_path), args)
