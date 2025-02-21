import nibabel
import nibabel.affines
import os
import numpy as np

# Custom modules
from vis import *

# Crop images to the minimum size whilst retaining whole dataset
# Can only be done on the whole dataset as the dataset has to be checked before
def crop_subjects(subject_list, relative_path, is_full_brain):
    
    max_bbox = (np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf)
    
    def bounding_box(image_array):
    
        non_zero_indices = np.nonzero(image_array)
        
        min_x, min_y, min_z = np.min(non_zero_indices[0]), np.min(non_zero_indices[1]), np.min(non_zero_indices[2])
        max_x, max_y, max_z = np.max(non_zero_indices[0]), np.max(non_zero_indices[1]), np.max(non_zero_indices[2])
        
        return (min_x, min_y, min_z, max_x, max_y, max_z)
    
    # Find maximum bbox
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
          
          
    for subject in subject_list:
        
        image = nibabel.load(os.path.join(subject.path, relative_path))
        
        image_array = image.get_fdata()
        
        global_min_x, global_min_y, global_min_z, global_max_x, global_max_y, global_max_z = max_bbox
        
        # Crop the image array using the global bounding box
        cropped_array = image_array[
            int(global_min_x):int(global_max_x),
            int(global_min_y):int(global_max_y),
            int(global_min_z):int(global_max_z)
        ]
        
        # Create a new NiBabel image from the cropped array
        cropped_image = nibabel.Nifti1Image(cropped_array, image.affine)
        
        display_image(cropped_image)
        
        fname = os.path.splitext(os.path.basename(relative_path))[0]
                
        cropped_path = os.path.join(subject.path,(fname + '_cropped.nii'))
        
        nibabel.save(cropped_image, cropped_path)
        
        if is_full_brain:
            
            subject.brain_aligned_cropped = cropped_path
            
        else:
            
            subject.aux_file_list.append(cropped_path)

            
    return
