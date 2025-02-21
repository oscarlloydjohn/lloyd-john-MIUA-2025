import nibabel
import nibabel.affines
from PIL import Image
import os
import fnmatch
import numpy as np
import ants
import concurrent.futures
import pandas as pd
import glob
import xml.etree.ElementTree as ET
import xmltodict
import shutil
import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Affine align a single subject
def alignment(subject):
        
    # Extract brain of subject and convert it to an ANTsPy image
    # The subject's brain is the moving image
    brain_array = extract_brain(subject.orig_nu, subject.mask)
    
    moving_image = ants.from_numpy(brain_array)
    
    # Convert the reference brain to an ANTsPy image
    # The reference brain is already extracted
    fixed_image = ants.from_numpy(reference_brain_array_fastsurfer)
    
    # Perform registration using ANTsPy
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='AffineFast')
    
    aligned_brain_array = registration['warpedmovout'].numpy()
    
    # Copy the temp mat transformation file to the subject directory
    shutil.copy(registration['fwdtransforms'][0], os.path.join(subject.path, 'affine_alignment.mat'))
    
    subject.affine_alignment = os.path.join(subject.path, 'affine_alignment.mat')
    
    # Make nibabel image from array
    # Identity matrix as affine transform
    aligned_image = nibabel.Nifti1Image(aligned_brain_array, np.eye(4))
    
    # Save the NiBabel image as a .nii file
    aligned_image_path = os.path.join(subject.path, 'brain_aligned.nii')
    
    nibabel.save(aligned_image, aligned_image_path)
    
    subject.brain_aligned = aligned_image_path
    
    return aligned_image

# Affine align a list of subjects in parallel 
def alignment_parallel(subject_list):
        
    # Use ProcessPoolExecutor to run affine alignment in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(alignment, subject))
        
        for future in concurrent.futures.as_completed(futures):
            
            display_image(future.result())
            
    return


# Uses affine_alignment.mat of a subject to align another file (for example the aseg file)
# Must have already been aligned using 
def aux_alignment(subject, file, is_aparc):
    
    # Open both images
    fixed_image = ants.from_numpy(np.asarray(nibabel.load(subject.brain_aligned).get_fdata()))
    
    moving_image = ants.from_numpy(np.asarray(nibabel.load(file).get_fdata()))
    
    # Must use nearest neighbours for interpolation to preserve discrete labels (colours), prevents blurring
    transformed_image = ants.apply_transforms(fixed_image, moving_image, subject.affine_alignment, interpolator='nearestNeighbor')
    
    ants.plot(fixed_image)
    
    ants.plot(transformed_image)
    
    path = os.path.join(subject.path,(os.path.splitext(os.path.basename(file))[0] + '_aligned.nii'))
    
    
    if is_aparc:
        
        # Convert to nibabel image
        # Make sure parcellation files are stored as int as they contain discrete values
        transformed_image = nibabel.Nifti1Image(transformed_image.numpy(), np.eye(4), dtype=np.int32)

        nibabel.save(transformed_image, path)
        
        subject.aparc_aligned = path
        
    else:
        
        # Convert to nibabel image
        transformed_image = nibabel.Nifti1Image(transformed_image.numpy(), np.eye(4))

        nibabel.save(transformed_image, path)
        
        subject.aux_file_list.append(path)
        
    
    return transformed_image

# Align aparc files in parallel
def aux_alignment_parallel(subject_list):
    
    # Use ProcessPoolExecutor to run affine alignment in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(aux_alignment, subject, subject.aparc, True))
            
        for future in concurrent.futures.as_completed(futures):
            
            display_image(future.result())
            
    return