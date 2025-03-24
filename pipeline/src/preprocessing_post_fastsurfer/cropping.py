import nibabel
import nibabel.affines
import os
import numpy as np
import concurrent.futures

# Custom modules
from .vis import *
from .subject import *

"""

Cropping
===========

This module provides functions for cropping of brain images to a bounding box. The functions are designed to be used with the Subject class. The bounding box is calculated across a list of subjects, where the maximum bounding box is found across all images such that all images can be cropped to the same size. This is useful when training a neural network on the dataset. 

Note that the max bounding box functions are not intended to be used before images have been aligned, as the bounding box will not tightly wrap these images due to their difference in position.

:author: Oscar Lloyd-John

"""

# Not for use with images of range greater than 0-255 as saves as uint8
def crop(subject: Subject, relative_path: os.PathLike[str], max_bbox: tuple[int, int, int, int, int, int], is_full_brain: bool = False):

    """

    Crops an image to the supplied bounding box which can be obtained using the get_max_bbox function. The cropped image is saved in the same directory as the original image with '_cropped' appended to the filename. If the image is a full brain, it is referred to in a subject attribute,

    :param subject: The subject containing the image to be cropped, or to store the cropped image in
    :type subject: Subject
    :param relative_path: The relative path to the image from the subject directory, usually will just be the filename
    :type relative_path: os.PathLike[str]
    :param max_bbox: The bounding box to crop the image to
    :type max_bbox: tuple[int, int, int, int, int, int]
    :param is_full_brain: Whether or not the image is a full brain, in which case it will be stored in the corresponding subject attribute, defaults to False
    :type is_full_brain: bool
    :return: A nibabel image of the cropped image
    :rtype: Nibabel image

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
def bounding_box(image_array: np.ndarray) -> tuple[int, int, int, int, int, int]:

    """

    Finds the bounding box of a 3d array, which is the smallest box that contains all non-zero elements of the array. The bounding box is returned as a tuple of the minimum and maximum x, y and z coordinates of the box.

    :param image_array: A 3d array representing an image
    :type image_array: np.ndarray
    :return: The bounding box of the image array
    :rtype: tuple[int, int, int, int, int, int]

    """

    non_zero_indices = np.nonzero(image_array)
    
    min_x, min_y, min_z = np.min(non_zero_indices[0]), np.min(non_zero_indices[1]), np.min(non_zero_indices[2])
    max_x, max_y, max_z = np.max(non_zero_indices[0]), np.max(non_zero_indices[1]), np.max(non_zero_indices[2])
    
    return (min_x, min_y, min_z, max_x, max_y, max_z)

def get_max_bbox(subject_list: list[Subject], relative_path: str) -> tuple[int, int, int, int, int, int]:

    """
    Finds the maximum bounding box of all images, such that all images can be cropped to the same size. The bounding box is returned as a tuple of the minimum and maximum x, y and z coordinates of the box. This bounding box is such it is as small as possible whilst ensuring that no image will have any non-zero data removed

    :param subject_list: A list of subjects to find the max bounding box of
    :type subject_list: list[Subject]
    :param relative_path: The relative path to the image from the subject directory, usually will just be the filename
    :type relative_path: str
    :return: The maximum bounding box of all images
    :rtype: tuple[int, int, int, int, int, int]
    """

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

def crop_subjects_parallel(subject_list: list[Subject], relative_path: os.PathLike[str], max_bbox: tuple[int, int, int, int, int, int], is_full_brain: bool = False, display: bool = False, display_3d: bool = False) -> None:

    """

    Crops a list of subjects to the same bounding box in parallel using a processpoolexecutor. The bounding box is passed in as an argument rather than being calculated in this function such that the bounding box can be calculated once and used across multiple calls to this function (in the case that the dataset is chunked).

    Simply runs crop on each subject in the list.

    :param subject_list: The list of subjects to crop
    :type subject_list: list[Subject]
    :param relative_path: The relative path to the image from the subject directory, usually will just be the filename
    :type relative_path: os.PathLike[str]
    :param max_bbox: The bounding box to crop the image to
    :type max_bbox: tuple[int, int, int, int, int, int]
    :param is_full_brain: Whether or not the image is a full brain, in which case it will be stored in the corresponding subject attribute, defaults to False
    :type is_full_brain: bool
    :param display: Whether or not to display the image upon cropping, defaults to False
    :type display: bool
    :param display_3d: Whether or not to display the image in 3d upon cropping, defaults to False
    :type display_3d: bool
    :return: None
    
    """
        
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
