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

    subjects = cohort_df['Subject'].unique()

    earliest_ids = []

    for subject in subjects:
        
        subject_df = cohort_df[cohort_df['Subject'] == subject]
        
        # Get the date that the earliest image was acquired
        earliest_date = subject_df.loc[[subject_df['Acq Date'].astype('datetime64[ns]').idxmin()]]['Acq Date'].values[0]
        
        # Filter to only those rows
        earliest_rows = subject_df[subject_df['Acq Date'] == earliest_date]
        
        # Randomly sample an image at the earliest date
        earliest_image = earliest_rows.sample()
        
        earliest_ids.append(earliest_image.iloc[0]['Image Data ID'])

    count = 0

    for item in os.listdir(data_path):
        
        item_path = os.path.join(data_path, item)
        
        image_id = item_path[item_path.rfind('_') + 1:item_path.rfind('.')]
        
        if image_id in earliest_ids:
            
            shutil.copy(item_path, os.path.join(dest_path, item))
            
            count += 1
        
    print(f"{count} images kept")