import os
import shutil
import argparse

"""
Directory Compaction Utility Script
=============================

A standalone script providing functions flatten a directory. All files in the directory and its subdirectories are moved to the top-level directory, and empty directories are removed.

Usage
-----

Run the script with the required arguments to delete files matching a specific filename recursively in a directory:

.. code-block:: bash

    python3 compact_dir.py --data_path /path/to/directory

:author: Oscar Lloyd-John
"""

# Recursively search all subdirectories for files and put them in data_path. Delete empty directories
def compact_dir(data_path: os.pathlike[str]) -> None:
    """
    Compacts the directory by moving all files from subdirectories to the top-level directory and removing any empty subdirectories.

    :param data_path: The path to the directory to be compacted.
    :type data_path: os.PathLike[str]
    :return: None
    """

    
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
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Flattens the given directory")
    
    parser.add_argument('--data_path', type=str, required=True)
    
    args = parser.parse_args()
    
    compact_dir(args.data_path)