import nibabel
import nibabel.affines
import os
import numpy as np
import ants
import concurrent.futures
import shutil

# Custom modules
from vis import *
from extraction import *

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

# Affine align a single subject
def alignment(subject, reference_brain):
        
    # Extract brain of subject and convert it to an ANTsPy image
    # The subject's brain is the moving image
    brain_array = extract_brain(subject.orig_nu, subject.mask)
    
    moving_image = ants.from_numpy(brain_array)
    
    # Convert the reference brain to an ANTsPy image
    # The reference brain is already extracted
    fixed_image = ants.from_numpy(reference_brain)
    
    # Perform registration using ANTsPy
    registration = ants.registration(fixed=fixed_image, moving=moving_image, type_of_transform='AffineFast')
    
    aligned_brain_array = registration['warpedmovout'].numpy()
    
    # Copy the temp mat transformation file to the subject directory
    shutil.copy(registration['fwdtransforms'][0], os.path.join(subject.path, 'affine_alignment.mat'))
    
    subject.affine_alignment = os.path.join(subject.path, 'affine_alignment.mat')
    
    # Make nibabel image from array
    # Identity matrix as affine transform
    aligned_image = nibabel.Nifti1Image(aligned_brain_array.astype('uint8'), np.eye(4))
    
    # Save the NiBabel image as a .nii file
    aligned_image_path = os.path.join(subject.path, 'brain_aligned.nii')
    
    nibabel.save(aligned_image, aligned_image_path)
    
    subject.brain_aligned = aligned_image_path
    
    return aligned_image

# Affine align a list of subjects in parallel 
def alignment_parallel(subject_list, reference_brain):
        
    # Use ProcessPoolExecutor to run affine alignment in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(alignment, subject, reference_brain))
        
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
        # int32 as aparc files can have values larger than 255
        transformed_image = nibabel.Nifti1Image(transformed_image.numpy().astype('uint32'), np.eye(4))

        nibabel.save(transformed_image, path)
        
        subject.aparc_aligned = path
        
    else:
        
        # Convert to nibabel image
        transformed_image = nibabel.Nifti1Image(transformed_image.numpy(), np.eye(4))

        nibabel.save(transformed_image, path)
        
        subject.aux_file_list.add(path)
        
    
    return transformed_image

# Align aparc files in parallel
def aux_alignment_parallel(subject_list, moving_image_attribute):
    
    # Use ProcessPoolExecutor to run affine alignment in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(aux_alignment, subject, getattr(subject, moving_image_attribute), True))
            
        for future in concurrent.futures.as_completed(futures):
            
            display_image(future.result())
            
    return
