import glob
import pandas as pd
from pprint import pprint
import os
import shutil

data_path = "/local/data/sc22olj/hcampus_data"
dest_path = "/local/data/sc22olj/isolated"

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