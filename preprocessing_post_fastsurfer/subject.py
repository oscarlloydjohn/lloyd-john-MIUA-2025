import os
import pandas as pd
import glob
from pprint import pprint
import concurrent.futures

# Really should be named Image, because a given subject can have multiple images and this only represents one of them
class Subject:
    
    # Constructor assumes that the directory has already been processed in the specific format using fastsurfer
    # See preprocess.py
    def __init__(self, path, subject_metadata):
        
        # Existing before object creation
        
        self.path = path
        
        self.orig_nu = os.path.join(path, "mri/orig_nu.mgz")
        
        self.mask = os.path.join(path, "mri/mask.mgz")
        
        self.aparc = os.path.join(path, "mri/aparc.DKTatlas+aseg.deep.mgz")
        
        self.subject_metadata = subject_metadata
    
            
        # Manually assign the column headers
        header = ['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName', 'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange']
        
        self.aseg_stats = pd.read_csv(os.path.join(path, 'stats/aseg+DKT.stats'), delimiter='\s+', comment='#', header=None, names=header)
        
        
        # Existing after object creation
        
        # Affine aligned brain
        brain_aligned = os.path.join(path, "brain_aligned.nii")
        
        self.brain_aligned = brain_aligned if os.path.isfile(brain_aligned) else None
        
        # Affine alignment matrix from ANTsPy
        affine_alignment = os.path.join(path, 'affine_alignment.mat')
        
        self.affine_alignment = affine_alignment if os.path.isfile(affine_alignment) else None
        
        # Aparc file aligned with matrix
        aparc_aligned = os.path.join(path, "aparc.DKTatlas+aseg.deep_aligned.nii")
        
        self.aparc_aligned = aparc_aligned if os.path.isfile(aparc_aligned) else None
        
        # Aligned and cropped brain
        brain_aligned_cropped = os.path.join(path, "brain_aligned_cropped.nii")
        
        self.brain_aligned_cropped = brain_aligned_cropped if os.path.isfile(brain_aligned_cropped) else None
        
        # NB specific regions e.g hippocampus are not stored in the object. Access them using their path directly
        
        # Set of all files for convenience
        self.aux_file_list = {f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))}
        
def get_cohort_df(data_path):
    
    csv_list = glob.glob(os.path.join(data_path, "*.csv"))

    # Read all CSVs into a list and concatenate
    cohort_df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)
    
    # Drop duplicate rows
    cohort_df = cohort_df.drop_duplicates(keep='first')
    
    # Find duplicate image IDs
    duplicates = cohort_df[cohort_df.duplicated(subset='Image Data ID', keep=False)]
    
    if not duplicates.empty:
        
        print("Error: duplicate image IDs in CSV")
        
        pprint(duplicates)
        
        return
    
    return cohort_df

# Given a subject directory, if it contains the correct structure then initialise an object
def init_subject(subject_path, cohort_df):
    
    def error(item: str):
        
        print(f"Subdirectory {item} does not match the expected format of a fastsurfer processed file. Check that processing was successful\n")
        
        return
        
    # MRI directory of subject path (checking validity)
    mri_path = os.path.join(subject_path, 'mri')
    
    # Check for MRI directory
    if os.path.isdir(mri_path):
        
        orig_file = os.path.join(mri_path, 'orig_nu.mgz')
        
        mask_file = os.path.join(mri_path, 'mask.mgz')

        # If both orig.mgz and mask.mgz exist, create object
        if os.path.isfile(orig_file) and os.path.isfile(mask_file):
            
            # Slice the string after the last underscore to get the image ID
            image_id = subject_path[subject_path.rfind('_') + 1:]
            
            # Get the subject's row using image ID
            subject_metadata = cohort_df.loc[cohort_df['Image Data ID'] == image_id].copy()
            
            if subject_metadata.empty:
                
                print("Error: Image ID not found")
                
                return
            
            return Subject(subject_path, subject_metadata)
        
    error(subject_path)
    
    return None

# Searches data_path for subject directories and creates an object for each of them
# NB subjects are actually 'images', there can be images for a given subject which are individual objects
def find_subjects_parallel(data_path):
    
    cohort_df = get_cohort_df(data_path)
    
    subject_list = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for item in os.listdir(data_path):
            
            subject_path = os.path.join(data_path, item)
            
            if os.path.isdir(subject_path):
                
                futures.append(executor.submit(init_subject, subject_path, cohort_df))
        
        for future in concurrent.futures.as_completed(futures):
            
            result = future.result()
            
            if result is not None:
                
                subject_list.append(result)
    
    return subject_list
        
        
# Searches data_path for subject directories and creates an object for each of them
# NB subjects are actually 'images', there can be images for a given subject which are individual objects
def find_subjects(data_path):
    
    cohort_df = get_cohort_df(data_path)
    
    subject_list = []
    
    for item in os.listdir(data_path):
        
        subject_path = os.path.join(data_path, item)
        
        if os.path.isdir(subject_path):
                    
            subject_list.append(init_subject(subject_path, cohort_df))

    return subject_list