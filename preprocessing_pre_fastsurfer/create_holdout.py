"""
Holdout set script
=============================

Stanalone script to move a subset of a directory to be used to test a final model. Automatically copies and CSVs into the target. Works with directories or single files 

Usage
-----

.. code-block:: bash

    python3 create_holdout.py --source /path/to/source --dest /path/to/dest --num <number of files>

:author: Oscar Lloyd-John
"""

import os
import shutil
import argparse
import random

def create_holdout(source, dest, num) -> None:

    """
    Creates a holdout set for machine learning

    """

    os.makedirs(dest, exist_ok=True)

    all_files = [file for file in os.listdir(source)]

    csvs = [file for file in all_files if file.endswith('.csv')]

    data = [file for file in all_files if not file.endswith('.csv')]

    files_to_move = random.sample(data, num)

    for file in files_to_move:

        shutil.move(os.path.join(source, file), os.path.join(dest, file))

    for csv in csvs:

        shutil.copy(os.path.join(source, csv), os.path.join(dest, csv))

    print(f"Moved {len(files_to_move)} files and copied {len(csvs)} csv(s) from {source} to {dest}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Creates a holdout set from the directory")
    
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--dest', type=str, required=True)
    parser.add_argument('--num', type=int, required=True)
    
    args = parser.parse_args()
    
    create_holdout(args.source, args.dest, args.num)