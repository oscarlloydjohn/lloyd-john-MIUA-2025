"""
ADNI download filtering script
=============================

A standalone script expecting a flattened directory of .nii files (most likely a flattened download from the ADNI image collections builder rather than the ARC builder). For each subject, the earliest images are identified and one image is randomly copied to a destination directory. This is usually the screening image. Image dates are identified by the csv from ADNI which should accompany the .nii files in the same directory. Images matched using their image IDs from the filenames.

The reason for this script is that the ADNI image collections builder does not provide a way to download only a single instance of a subject's screening images. It is necessary to have only one image per subject for training a neural network.

Usage
-----

Run the script with the required arguments to delete files matching a specific filename recursively in a directory:

.. code-block:: bash

    python3 isolate_adni_screening.py --data_path /path/to/directory --dest_path /path/to/destination

:author: Oscar Lloyd-John
"""

import glob
import pandas as pd
import os
import shutil
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="For all subjects, find their earliest image and copy it to the destination. Usually these are the screening images, hence the name")
    
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dest_path', type=str, required=True)
    
    args = parser.parse_args()

    data_path = args.data_path
    dest_path = args.dest_path

    # For all images of a given subject at the earliest visit, chooses one random image from that date and copies that to dest
    csv_list = glob.glob(os.path.join(data_path, "*.csv"))

    cohort_df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)

    subjects = cohort_df['Subject ID'].unique()

    earliest_ids = []

    for subject in subjects:
        
        subject_df = cohort_df[cohort_df['Subject ID'] == subject]
        
        # Get the date that the earliest image was acquired
        earliest_date = subject_df.loc[[subject_df['Study Date'].astype('datetime64[ns]').idxmin()]]['Study Date'].values[0]
        
        # Filter to only those rows
        earliest_rows = subject_df[subject_df['Study Date'] == earliest_date]
        
        # Randomly sample an image at the earliest date
        earliest_image = earliest_rows.sample()
        
        earliest_ids.append(earliest_image.iloc[0]['Image ID'])

    count = 0

    for item in os.listdir(data_path):
        
        item_path = os.path.join(data_path, item)
        
        image_id = item_path[item_path.rfind('_') + 2 :item_path.rfind('.')]
        
        if image_id in earliest_ids:
            
            shutil.copy(item_path, os.path.join(dest_path, item))
            
            count += 1
        
    print(f"{count} images kept")