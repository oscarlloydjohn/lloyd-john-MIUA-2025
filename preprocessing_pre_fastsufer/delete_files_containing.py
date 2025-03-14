import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import argparse

"""
File Deletion Utility Script 2
=============================

A standalone script providing functions to delete files in a directory that contain a matching string in their filename. Deletions are performed in parallel using a ThreadPoolExecutor.

Usage
-----

.. code-block:: bash

    python3 delete_files_containing.py --data_path /path/to/directory --string scaled

"""

def delete_files_containing(data_path: os.PathLike[str], string: str) -> None:

    """
    Similar to delete_files_in_directory, but removes files containing the specified string in the filename rather than the exact filename. Works in parallel

    :param data_path: The path to the directory containing the files.
    :type data_path: os.pathlike[str]
    :param string: The string to search for in the file names.
    :type string: str
    :returns: None
    """
    
    files = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    
    files_to_remove = [f for f in files if string in f]
    
    def delete_file(file_name):
        
        try:
            
            file_path = os.path.join(data_path, file_name)
            
            os.remove(file_path)
            
            print(f"Deleted: {file_name}")
            
        except Exception as e:
            
            print(f"Error deleting {file_name}: {e}")

    with ThreadPoolExecutor() as executor:
        
        executor.map(delete_file, files_to_remove)

    return

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Delete all files that have a matching string in their filename, in the given directory (non-recursive)")
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--string', type=str, required=True)
    
    args = parser.parse_args()
    
    delete_files_containing(args.data_path, [args.string])