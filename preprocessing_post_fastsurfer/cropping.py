import nibabel
import nibabel.affines
import os
import numpy as np
import concurrent.futures

# Custom modules
from .vis import *

# Not for use with images of range greater than 0-255 as saves as uint8
def crop(subject, relative_path, max_bbox, is_full_brain):
    """

    Args:
        subject (_type_): _description_
        relative_path (_type_): _description_
        max_bbox (_type_): _description_
        is_full_brain (bool): _description_

    Returns:
        _type_: _description_
    """
    
    image = nibabel.load(os.path.join(subject.path, relative_path))
    
    image_array = image.get_fdata()
    
    min_x, min_y, min_z, max_x, max_y, max_z = max_bbox
    
    # Crop the image array using the global bounding box
    cropped_array = image_array[
        int(min_x):int(max_x),
        int(min_y):int(max_y),
        int(min_z):int(max_z)
    ]
    
    # Create a new NiBabel image from the cropped array
    cropped_image = nibabel.Nifti1Image(cropped_array.astype('uint8'), image.affine)
    
    fname = os.path.splitext(os.path.basename(relative_path))[0]
            
    cropped_path = os.path.join(subject.path,(fname + '_cropped.nii'))
    
    nibabel.save(cropped_image, cropped_path)
    
    if is_full_brain:
        
        subject.brain_aligned_cropped = cropped_path
        
    else:
        
        subject.aux_file_list.add(cropped_path)
    
    return cropped_image

# Finds the bounding box of the image data
def bounding_box(image_array):

    non_zero_indices = np.nonzero(image_array)
    
    min_x, min_y, min_z = np.min(non_zero_indices[0]), np.min(non_zero_indices[1]), np.min(non_zero_indices[2])
    max_x, max_y, max_z = np.max(non_zero_indices[0]), np.max(non_zero_indices[1]), np.max(non_zero_indices[2])
    
    return (min_x, min_y, min_z, max_x, max_y, max_z)

def get_max_bbox(subject_list, relative_path):
    
    max_bbox = (np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf)
    
    # Find optimal bbox in serial
    for subject in subject_list:
    
        image = nibabel.load(os.path.join(subject.path, relative_path))
        
        min_x, min_y, min_z, max_x, max_y, max_z = bounding_box(image.get_fdata())
        
        # Update max_bbox
        max_bbox = (
            min(max_bbox[0], min_x),
            min(max_bbox[1], min_y),
            min(max_bbox[2], min_z),
            max(max_bbox[3], max_x),
            max(max_bbox[4], max_y),
            max(max_bbox[5], max_z)
        )
        
    return max_bbox

# Crop images to the minimum size whilst retaining whole dataset
# Can only be done on the whole dataset as the dataset has to be checked before
# Cropping is performed in parallel
def crop_subjects_parallel(subject_list, relative_path, max_bbox, is_full_brain=False, display=False, display_3d=False):
        
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(crop, subject, relative_path, max_bbox, is_full_brain))
            
        for future in concurrent.futures.as_completed(futures):
            
            if display_3d:
                
                display_image_3d(future.result(), 7, mode='preview')
            
            if display:
                
                display_image(future.result())
            
    return
