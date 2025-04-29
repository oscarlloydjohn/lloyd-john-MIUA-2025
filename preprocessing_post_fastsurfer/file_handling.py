"""
File handling
===========

This module provides simple functions for handling files when processing brain images. The functions are designed to be used with the Subject class.

:author: Oscar Lloyd-John
"""

import os
import fnmatch

from .subject import Subject

def list_files_ext(data_path: os.PathLike[str], extensions: str) -> list:

    """

    List all files in a directory with a specified extension

    :param data_path: The dataset directory to be searched
    :type data_path: os.PathLike[str]
    :param extensions: The file extension to search for
    :type extensions: str

    :return: List of absolute file paths to the files in the directory with the specified extensions
    :rtype: list

    """

    files = [f for f in os.listdir(data_path) if f.endswith(extensions)]
        
    return files       

def list_files_fname(data_path: os.PathLike[str], filename: str) -> list:

    """

    Lists all files in a directory with a specified filename (recursive). Useful for clearing up intermediate files from processing.

    :param data_path: The dataset directory to be searched
    :type data_path: os.PathLike[str]
    :param filename: The filename to search for
    :type filename: str
    :return: List of absolute file paths to the files in the directory with the specified filename
    :rtype: list

    """
    
    matched_files = []
    
    for root, dirs, files in os.walk(data_path):
        
        for file in fnmatch.filter(files, filename):
            
            matched_files.append(os.path.join(root, file))
    
    return matched_files

# Delete the files returned by list_files_fname
def delete_file_matching(data_path: os.PathLike[str], filename: str) -> None:

    """
    
    Deletes all files in a directory with a specified filename (recursive). Useful for clearing up intermediate files from processing. Simply deletes all files returned by list_files_fname.

    :param data_path: The dataset directory to be searched
    :type data_path: os.PathLike[str]
    :param filename: The filename to search for and delete
    :type filename: str
    :return: None

    """
    
    for file in list_files_fname(data_path, filename):
        
        os.remove(file)
    
    return

def delete_aux_files(subject: Subject) -> None:

    """
    Deletes all files contained in the aux_file_list subject attribute. Useful for clearing up intermediate files from processing. Will delete files that are stored in another class attribute if they have been added to aux_file_list.

    :param subject: The subject to delete the auxiliary files from
    :type subject: Subject
    :return: None

    """
    
    for file in subject.aux_file_list[:]:
        
            subject.aux_file_list.remove(file)
        
            os.remove(os.path.normpath(file))

    return