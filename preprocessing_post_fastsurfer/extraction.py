import nibabel
import nibabel.affines
import os
import numpy as np
import pandas as pd
import concurrent.futures

# Custom modules
from .vis import *

# Extracts brain regions using their number label found from freesurfer LUT
def extract_region(subject, values_list, brain, aparc, is_aligned=False):
    
    # Doesn't use array proxies as proxies cast to float
    image_array = nibabel.load(brain).dataobj
    
    # Must be int for comparison with values list. Cannot cast using numpy due to errors
    # NB aparcs must be saved as int files
    aparc_array = nibabel.load(aparc).dataobj
    
    # Create a mask from regions in list
    filtered_array = np.where(np.isin(aparc_array, values_list), 1, 0)
    
    # Check for empty array
    if np.all(filtered_array == 0):
        
        print("Error: region empty")
        
        return filtered_array
    
    # Extract region using mask
    extracted_region = image_array * filtered_array

    # Look up the name of the region for the filename
    lut_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/preprocessing_post_fastsurfer/FreeSurferColorLUT.txt"
    
    lut = pd.read_csv(lut_path, delimiter='\s+', comment='#', header=None)
    
    region_names = lut[lut[0].isin(values_list)][1]

    # Save the regions as a nii file, casting to uint8 for compatability and size
    region_image = nibabel.Nifti1Image(extracted_region.astype('uint8'), np.eye(4))
    
    if is_aligned:
            
        region_image_path = os.path.join(subject.path, ('_'.join(region_names) + '_aligned.nii'))
        
    else:
            
        region_image_path = os.path.join(subject.path, ('_'.join(region_names) + '.nii'))
        
    nibabel.save(region_image, region_image_path)
        
    subject.aux_file_list.add(region_image_path)
    
    return region_image

def extract_region_parallel(subject_list, values_list, brain_attribute, aparc_attribute, is_aligned, display=False, display_3d=False):
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(extract_region, subject, values_list, getattr(subject, brain_attribute), getattr(subject, aparc_attribute), is_aligned))
            
        for future in concurrent.futures.as_completed(futures):
            
            if display_3d:
                
                display_image_3d(future.result(), 7, mode='preview')
            
            if display:
                
                display_image(future.result())
            
    return