import os
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor
import argparse

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
        f"--sd", data_path,
        "--sid", dirname,
        f"--t1", f"{data_path}/{filename}",
        "--seg_only", "--threads", f"{threads}", "--device", "cpu"
    ]

    process = subprocess.Popen(command)
    
    process.wait()
    
    return
