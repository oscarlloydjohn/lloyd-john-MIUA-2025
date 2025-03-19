
# Torch
import torch
from torch.utils.data import Dataset
import torch
from torcheval.metrics import *

# Other
import nibabel
import os
import numpy as np
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *

"""

Subject dataset
===========

:author: Oscar Lloyd-John

"""

class SubjectDataset(Dataset):

    """
    This is a class which inherits from the PyTorch Dataset class. It is designed to be used with the Subject class from preprocessing_post_fastsurfer, effectively it wraps that class. Allows for configuration of the MRI dataset for use in pytorch.

    Has an auxilliary function load_mri_to_tensor which is used to load MRI data from a file path to a tensor. Point clouds and meshes are easily loaded from numpy.

    This class allows the dataset to be loaded into memory or read from disk per subject (when the dataloader needs it). Reading from disk can be useful for prototyping however can cause bottlenecking issues when using a network drive whilst training many models on the same data. For a dataset of approx 500 subjects, loading voxel or pointcloud data into memory is fine if you have maybe 32gb of RAM. Don't let it go into swap though.

    Parameters:
    ----------
    :param data_path: The path to the dataset directory as used by init_subjects
    :type path: os.PathLike[str]
    :param selected_labels: The labels to be used in the dataset, must be 2 or 3 labels, which are a combination of 'CN', 'MCI' and 'AD'. Must start with 'CN' as this is assumed to be class 0
    :type selected_labels: list[str]
    :param load_in_memory: Whether to load the entire dataset into memory or read from disk per subject
    :type load_in_memory: bool
    :param data_string: The key string for the data in the dictionary i.e which file are we interested in from the subjects directory
    :type data_string: str
    :param labels_string: The key string for the labels in the dictionary. In this case will be research_group unless modified

    Attributes:
    ----------
    :ivar load_in_memory: Whether to load the entire dataset into memory
    :ivar selected_labels: The labels to be used in the dataset
    :ivar num_classes: The number of classes, derived from the number of selected labels
    :ivar subject_list: The list of subjects that we want to include in the dataset, returned by find_subjects_parallel
    :ivar mem_dict: The dictionary of actual data and labels for each subject, if load_in_memory is True
    

    :author: Oscar Lloyd-John 
    :contact: sc22olj@leeds.ac.uk
    """
    
    def __init__(self, data_path, selected_labels, load_in_memory=False, data_string=None, labels_string='research_group'):

        """
        Initialises the dataset

        """
        
        # Whether to load the entire dataset into memory
        self.load_in_memory = load_in_memory

        self.data_string = data_string

        self.selected_labels = selected_labels
        
        self.num_classes = len(selected_labels)
        
        # Check format of selected labels
        if len(selected_labels) < 2 or len(selected_labels) > 3:
            
            print("Error, must have 2 or 3 labels \n")
            
            return
        
        if selected_labels[0] != 'CN':
            
            print("Error, ordering of labels incorrect")
            
            return
        
        
        initial_subject_list = find_subjects_parallel(data_path)
        
        final_subject_list = []
            
        # Include only the subjects that have a label of interest
        # Separate these into individual lists
        label_subgroups = {label: [] for label in selected_labels}
            
        for subject in initial_subject_list:
            
            label = subject.subject_metadata['Research Group'].iloc[0]
            
            if label in selected_labels:
                
                label_subgroups[label].append(subject)
        
        # Append the subject list with each label's subgroup
        for label in selected_labels:
            
            final_subject_list.extend(label_subgroups[label])
            
        # Shuffle in case of ordering in directory
        random.shuffle(final_subject_list)
        
        self.subject_list = final_subject_list
        
        subject.mem_dict = None
        
        # Load all the data into the dataset rather than only per subject when needed
        if load_in_memory:
            
            print("Loading dataset into memory\n")
            
            for index, subject in tqdm(enumerate(self.subject_list), total=len(self.subject_list)):
                
                subject.mem_dict = self.load_subject(index)
            
    def __len__(self):
        
        return len(self.subject_list)

    def __getitem__(self, index):
        
        if self.load_in_memory:
            
            subject = self.subject_list[index]
            
            return subject.mem_dict
        
        else:
            
            return self.load_subject(index)
        
    def load_subject(self, index: int) -> dict:
        """
        Loads the data from the file paths and dataframes in the Subject object

        :param index: The index of the subject in the subject list
        :type index: int
        :return: A dictionary containing the data and labels for the subject
        :rtype: dict
        """
        
        subject = self.subject_list[index]
        data = {}

        """IMAGES"""
        try:
            if 'brain' in self.data_string:
                data['brain'] = self.load_mri_to_tensor(subject.brain_aligned)
            
            if 'brain_cropped' in self.data_string:
                data['brain_cropped'] = self.load_mri_to_tensor(subject.brain_aligned_cropped)
            
            if 'hcampus_vox' in self.data_string:
                data['hcampus_vox'] = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_cropped.nii'))
            
            if 'hcampus_pointcloud' in self.data_string:
                data['hcampus_pointcloud'] = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_cropped_mesh_downsampledcloud.npy')), dtype=torch.float32)
            
            if 'lhcampus_vox' in self.data_string:
                data['lhcampus_vox'] = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_cropped.nii'))
            
            if 'lhcampus_pointcloud' in self.data_string:
                data['lhcampus_pointcloud'] = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_cropped_mesh_downsampledcloud.npy')), dtype=torch.float32)
            
            if 'hcampus_vox_aligned' in self.data_string:
                data['hcampus_vox_aligned'] = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_aligned_cropped.nii'))
            
            if 'hcampus_pointcloud_aligned' in self.data_string:
                data['hcampus_pointcloud_aligned'] = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_aligned_cropped_mesh_downsampledcloud.npy')), dtype=torch.float32)
            
            if 'lhcampus_vox_aligned' in self.data_string:
                data['lhcampus_vox_aligned'] = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_aligned_cropped.nii'))
            
            if 'lhcampus_pointcloud_aligned' in self.data_string:
                data['lhcampus_pointcloud_aligned'] = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_aligned_cropped_mesh_downsampledcloud.npy')), dtype=torch.float32)
        
        except Exception as e:
            print(f"Error loading images, make sure all files associated with given keys are available: {e}")

        """REGION VOLUME STATS"""
        if 'volumes' in  self.data_string:

            volume_col = subject.aseg_stats['Volume_mm3']
            volume_col_normalised = volume_col / volume_col.sum() * 1000
            struct_name_col = subject.aseg_stats['StructName']
            
            data['volumes'] = np.array(volume_col_normalised)
            
            data['struct_names'] = np.array(struct_name_col)

        """SUBJECT INFO - NB NOT COMPLETE WITH SCORES"""
        if 'scores' in self.data_string:
            subject_metadata = subject.subject_metadata
            
            mmse = subject_metadata['MMSE Total Score'].iloc[0]
            gdscale = subject_metadata['GDSCALE Total Score'].iloc[0]
            faq = subject_metadata['FAQ Total Score'].iloc[0]
            npiq = subject_metadata['NPI-Q Total Score'].iloc[0]

            data['scores'] = [mmse, gdscale, faq, npiq]
        
            data['score_names'] = ['MMSE Total Score', 'GDSCALE Total Score', 'FAQ Total Score', 'NPI-Q Total Score']
        
            # Convert research group disease label str to number for pytorch
            if len(self.selected_labels) == 3:
                mapping = {
                    'CN': 0,
                    'MCI': 1,
                    'AD': 2
                }
            elif len(self.selected_labels) == 2:
                mapping = {
                    'CN': 0,
                    'MCI': 1,
                    'AD': 1
                }

            # Get the value of the mapping, -1 if not found
            research_group = mapping.get(subject_metadata['Research Group'].iloc[0], -1)
            data['research_group'] = research_group

        return data
        
    def load_mri_to_tensor(self, path: os.PathLike[str]) -> torch.Tensor:

        """

        Loads an mri file, probably an .mgz or nii file. Likely to be called using a Subject object attribute

        :param path: The path to the mri file
        :type path: os.PathLike[str]
        :return: The mri data as a tensor
        :rtype: torch.Tensor

        """
        
        if path is None or not os.path.isfile(path):
            return torch.empty(0)  # Return empty tensor if the file doesn't exist
        
        # Example of using nibabel to load .mgz files (you can modify as needed)
        image = nibabel.load(path)

        image_data = image.get_fdata()
        
        # Convert to PyTorch tensor
        tensor_data = torch.tensor(image_data, dtype=torch.float32)
        
        return tensor_data