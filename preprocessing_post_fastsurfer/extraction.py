"""

Extraction
===========

This module provides functions extracting brain regions from a brain image using the freesurfer parcellation files and the corresponding lookup table. The functions are designed to be used with the Subject class.

:author: Oscar Lloyd-John

"""

import nibabel
import nibabel.affines
import os
import numpy as np
import pandas as pd
import concurrent.futures

# Custom modules
from .vis import *
from .subject import *

def extract_region(subject: Subject, values_list: list[int], brain: os.PathLike[str], aparc: os.PathLike[str], is_aligned: bool = False):

    """

    Extracts one or more regions from a brain image using the freesurfer parcellation file. The desired regions are specified by a list of values corresponding to the region number LUT file. The region is extracted by element-wise multiplication of the brain image and a mask created from the parcellation file (a region of the parcellation file is isolated based on the input values).
    The resulting filename is the region name(s) followed by .nii or _aligned.nii depending on whether the region is aligned or not. See the FreeSurferColourLUT.txt file for region names and their corresponding values.

    Extracted regions are saved in the subject directory and in the aux_file_list but are not stored as subject attributes as there are too many possible regions to store.

    NB the lookup table is an absolute path at the moment, this needs to be changed

    :param subject: The subject containing the image to be extracted
    :type subject: Subject
    :param values_list: The list of region values to be extracted, found from the FreeSurferColorLUT.txt file
    :type values_list: list[int]
    :param brain: The path to the brain image to be extracted from
    :type brain: os.PathLike[str]
    :param aparc: The path to the aparc file to be used as a mask, make sure this is aligned with the brain image
    :type aparc: os.PathLike[str]
    :param is_aligned: Whether or not the image is aligned, defaults to False
    :type is_aligned: bool
    :return: A nibabel image of the extracted region
    :rtype: Nibabel image

    """
    
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
    lut_path = os.path.join(os.path.dirname(__file__), "FreeSurferColorLUT.txt")
    
    lut = pd.read_csv(lut_path, delimiter=r'\s+', comment='#', header=None)
    
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

def extract_region_parallel(subject_list: list[Subject], values_list: list[int], brain_attribute: str, aparc_attribute: str, is_aligned: bool, display: bool = False, display_3d: bool = False) -> None:

    """
    
    Extracts region(s) from a list of subjects in parallel using a processpoolexecutor. Simply calls extract_region on each subject in the list. This does not work with filenames, only Subject attributes

    :param subject_list: The list of subjects to have regions extracted
    :type subject_list: list[Subject]
    :param values_list: The list of region values to be extracted, found from the FreeSurferColorLUT.txt file
    :type values_list: list[int]
    :param brain_attribute: A string representation of the Subject attribute referring to the brain image to be extracted from
    :type brain_attribute: str
    :param aparc_attribute: A string representation of the Subject attribute referring to the aparc image to be used as a mask
    :type aparc_attribute: str
    :param is_aligned: Whether or not the image is aligned, defaults to False
    :type is_aligned: bool
    :param display: Whether or not to display the image upon extraction, defaults to False
    :type display: bool
    :param display_3d: Whether or not to display the image in 3D upon extraction, defaults to False
    :type display_3d: bool
    :return: None

    """
    
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