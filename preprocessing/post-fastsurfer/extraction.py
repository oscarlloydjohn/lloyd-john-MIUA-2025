import nibabel
import nibabel.affines
import os
import numpy as np
import pandas as pd

# Performs brain extraction using the orig_nu.mgz and mask.mgz of the subject by multiplying the mask with the image
def extract_brain(orig_file, mask_file):
    
    # Load the image and the brain mask
    image = nibabel.load(orig_file)
    mask = nibabel.load(mask_file)
    
    # Get their image arrays
    image_array = np.asarray(image.dataobj)
    mask_array = np.asarray(mask.dataobj)
    
    # Apply the mask, the mask entries are 1 or 0
    brain_array = image_array * mask_array
    
    return brain_array

# Extracts brain regions using their number label found from freesurfer LUT
# Takes regions as a name
def extract_region(subject, values_list, brain, aparc, is_aligned):
    
    aparc_array = nibabel.load(aparc).get_fdata()
    
    image_array = nibabel.load(brain).get_fdata()
    
    # Create a mask from regions in list
    filtered_array = np.where(np.isin(aparc_array, values_list), 1, 0)
    
    # Check for empty array
    if np.all(filtered_array == 0):
        
        print("Error: region empty")
        
        return filtered_array

    
    # Extract region using mask
    extracted_region = image_array * filtered_array
    
    # Look up the name of the region for the filename
    lut_path = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/preprocessing/FreeSurferColorLUT.txt"
    
    lut = pd.read_csv(lut_path, delimiter='\s+', comment='#', header=None)
    
    region_names = lut[lut[0].isin(values_list)][1]

    # Save the regions as a nii file
    region_image = nibabel.Nifti1Image(extracted_region, np.eye(4))
    
    if is_aligned:
            
        region_image_path = os.path.join(subject.path, ('_'.join(region_names) + '_aligned.nii'))
        
    else:
            
        region_image_path = os.path.join(subject.path, ('_'.join(region_names) + '.nii'))
        
    nibabel.save(region_image, region_image_path)
        
    subject.aux_file_list.append(region_image_path)
    
    return extracted_region