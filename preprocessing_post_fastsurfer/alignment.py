"""
Alignment
===========

This module provides functions for brain extraction and affine alignment of both the brain and auxilliary files such as parcellation files. The functions are designed to be used with the Subject class. 

Because Nibabel does not have affine alignment, and ANTsPy does not have support for .mgh/.mgz files, the functions in this module use a hybrid of both libraries.

:author: Oscar Lloyd-John
"""

import nibabel
import nibabel.affines
import os
import numpy as np
import ants
import concurrent.futures
import shutil

# Custom modules
from .vis import *
from .extraction import *
from .subject import *

def extract_brain(orig_file: os.PathLike[str], mask_file: os.PathLike[str]) -> np.ndarray:
    """
    Given a brain image and a mask, extracts the brain from the image by element-wise multiplication of the image and the mask. The mask is assumed to be binary, with 1s representing brain and 0s representing non-brain.

    :param orig_file: The path to the brain MRI, likely the orig_nu.mgz file.
    :type orig_file: os.PathLike[str]
    :type mask_file: The path to the mask file, likely the mask.mgz file.
    :type mask_file: os.PathLike[str]
    :returns: A 3D array representing the extracted brain
    :rtype: np.ndarray

    """  

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
def alignment(subject: Subject, reference_brain: np.ndarray):

    """
    Aligns the brain (orig_nu.mgz) of a subject to a reference brain using ANTsPy, however loads images in using Nibabel. The ANTsPy affine transformation file is saved for use in aligning auxilliary files. Returns a nibabel image of the aligned brain.

    The reference brain should be an average brain such as the MNI template brain, that has been processed in the same way as the subject dataset using Fastsurfer. The brain is then extracted using extract_brain before passing to this function.

    :param subject: The subject to be aligned
    :type subject: Subject
    :param reference_brain: A 3d array representing the reference brain to align the subject to
    :returns: A nibabel image of the aligned brain
    :rtype: Nibabel image
    """
        
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
def alignment_parallel(subject_list: list[Subject], reference_brain: np.ndarray, display: bool = False, display_3d: bool = False) -> None:
    """
    Performs affine alignment of orig_nu.mgz in parallel using a processpoolexecutor. The reference brain should be an average brain such as the MNI template brain, that has been processed in the same way as the subject dataset using Fastsurfer. The brain is then extracted using extract_brain before passing to this function.

    Display arguments call the functions from vis.

    :param subject_list: A list of Subject objects
    :type subject_list: list[Subject]
    :param reference_brain: A 3d array representing the reference brain
    :type reference_brain: np.ndarray
    :param display: Display the middle slice of the aligned brain, defaults to False
    :type display: bool, optional
    :param display_3d: Display the aligned brain in 3d using pyvista, defaults to False
    :type display_3d: bool, optional
    """
        
    # Use ProcessPoolExecutor to run affine alignment in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(alignment, subject, reference_brain))
        
        for future in concurrent.futures.as_completed(futures):
            
            if display_3d:
                
                display_image_3d(future.result(), 7, mode='preview')
            
            if display:
                
                display_image(future.result())
                
    return

def aux_alignment(subject: Subject, file: str, is_aparc: bool = False):
    """
    Uses the affine_alignment.mat file from alignment to align another file to the aligned brain. The idea is that this can be used to align parcellation files or masks such that regions can be extracted from the aligned brain easily. Requires that the subject has already had the main brain aligned.

    If the is_aparc flag is set to true, the aligned path is stored in the Subject's aparc_aligned attribute. Note that if the flag is true, uint32 is used as the datatype as aparc files can have values larger than 255. Otherwise, it is assumed that the image is a normal grayscale MRI.

    :param subject: The subject to have aux file aligned
    :type subject: Subject
    :param file: The path to the file to be aligned, likely a parcellation file
    :type file: str
    :param is_aparc: Whether or not the file is the aparc attribute of the subject, defaults to False
    :type is_aparc: bool, optional
    :return: A Nibabel image of the aligned file
    :rtype: Nibabel image
    """
    
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
        transformed_image = nibabel.Nifti1Image(transformed_image.numpy().astype('uint8'), np.eye(4))

        nibabel.save(transformed_image, path)
        
        subject.aux_file_list.add(path)
        
    
    return transformed_image

# Align aparc files in parallel
def aux_alignment_parallel(subject_list: list[Subject], moving_image_attribute, is_aparc=False, display=False):
    """
    Performs auxiliary file alignment in parallel using a processpoolexecutor. Simply calls aux_alignment on each subject in the list.

    :param subject_list: The list of subjects to have aux files aligned
    :type subject_list: list[Subject]
    :param moving_image_attribute: The Subject attribute to be aligned, likely the aparc attribute
    :type moving_image_attribute: str
    :param is_aparc: Whether or not the file to be aligned is an aparc, in which case it will be stored in uint32, defaults to False
    :type is_aparc: bool, optional
    :param display: Whether or not to display the image upon alignment, defaults to False
    :type display: bool, optional
    :return: None
    """


    # Use ProcessPoolExecutor to run affine alignment in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(aux_alignment, subject, getattr(subject, moving_image_attribute), is_aparc))
            
        for future in concurrent.futures.as_completed(futures):
            
            if display:
            
                display_image(future.result())
            
    return
