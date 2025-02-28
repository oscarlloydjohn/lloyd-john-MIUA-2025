import os
import pandas as pd
import glob
import xmltodict
from pprint import pprint

class Subject:
    
    # Constructor assumes that the directory has already been processed in the specific format using fastsurfer
    # See preprocess.py
    def __init__(self, path, metadata_df):
        
        # Existing before object creation
        self.path = path
        
        self.orig_nu = os.path.join(path, "mri/orig_nu.mgz")
        
        self.mask = os.path.join(path, "mri/mask.mgz")
        
        self.aparc = os.path.join(path, "mri/aparc.DKTatlas+aseg.deep.mgz")
        
        
        
        xml_files = glob.glob(os.path.join(path, "*.xml"))
        
        self.xml_path = xml_files[0] if xml_files else None
        
        with open(self.xml_path, 'r') as file:
            
                self.xml_df = xmltodict.parse(file.read())

        # Manually assign the column headers
        header = ['ColHeaders', 'Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName', 'normMean', 'normStdDev', 'normMin', 'normMax', 'normRange']
        
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
        
        # NB specific regions e.g hippocampus are not stored in the object. Access them using their path from the aux file list
        
        # Set of all files for convenience
        self.aux_file_list = {f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))}
        
        
# Searches data_path for subject directories and creates an object for each of them
def find_subjects(data_path):
    
    subject_list = []
    
    csv_list = glob.glob(os.path.join(data_path, "*.csv"))

    # Read all CSVs into a list and concatenate
    metadata_df = pd.concat([pd.read_csv(csv) for csv in csv_list], ignore_index=True)
    
    # Find and print duplicates, drop the first
    duplicates = metadata_df[metadata_df.duplicated(subset='Image Data ID', keep=False)]
    
    if not duplicates.empty:
        
        pprint(duplicates)
        
        metadata_df = metadata_df.drop_duplicates(subset='Image Data ID', keep='first')
    
    for item in os.listdir(data_path):
        
        subject_path = os.path.join(data_path, item)
        
        if os.path.isdir(subject_path):
            
            # MRI directory of subject path (checking validity)
            mri_path = os.path.join(subject_path, 'mri')
            
            # Check for MRI directory
            if os.path.isdir(mri_path):
                
                orig_file = os.path.join(mri_path, 'orig_nu.mgz')
                
                mask_file = os.path.join(mri_path, 'mask.mgz')

                # If both orig.mgz and mask.mgz exist, create object
                if os.path.isfile(orig_file) and os.path.isfile(mask_file):
                    
                    print(subject_path[-6])
                    
                    subject_list.append(Subject(subject_path, metadata_df))

    return subject_list