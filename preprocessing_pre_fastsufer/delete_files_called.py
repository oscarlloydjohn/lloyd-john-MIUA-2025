import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import argparse

"""
File Deletion Utility Script 1
=============================

A standalone script providing functions to delete files in a directory that match a filename. Deletions are performed in parallel using a ThreadPoolExecutor.

Usage
-----

.. code-block:: bash

    python3 delete_files_called.py --data_path /path/to/directory --filename file1.txt

"""


def delete_file(file_path: os.PathLike[str]) -> None:
    """
    Deletes a single file at the given path.

    :param file_path: The path to the file to be deleted.
    :type file_path: os.PathLike[str]
    :returns: None
    """
    
    try:
        os.remove(file_path)
        
        print(f"Deleted: {file_path}")
        
    except Exception as e:
        
        print(f"Failed to delete {file_path}: {e}")

    return

def delete_files_in_directory(data_path: os.pathlike[str], filenames_to_delete: list[str]) -> None:
    """
    Deletes files with the specified filenames in the given directory and all subdirectories. Works in parallel

    :param data_path: The path to the directory where the search for files will begin.
    :type data_path: os.PathLike[str]
    :param filenames_to_delete: A list of filenames to be deleted
    :type filenames_to_delete: list[str]
    :returns: None
    """

    files_to_delete = []

    for dirpath, _, filenames in os.walk(data_path):
        
        for filename in filenames:
            
            if filename in filenames_to_delete:
                
                files_to_delete.append(os.path.join(dirpath, filename))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        
        executor.map(delete_file, files_to_delete)

    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Delete all files matching the filename in the given directory (recursive)")
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)
    
    args = parser.parse_args()
    
    delete_files_in_directory(args.data_path, [args.filename])
