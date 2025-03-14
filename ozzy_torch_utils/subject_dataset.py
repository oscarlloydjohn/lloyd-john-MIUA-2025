
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

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *

class SubjectDataset(Dataset):
    def __init__(self, data_path, selected_labels, downsample_majority=True):
        
        # Check format of selected labels
        if len(selected_labels) < 2 or len(selected_labels) > 3:
            
            print("Error, must have 2 or 3 labels \n")
            
            return
        
        if selected_labels[0] != 'CN':
            
            print("Error, ordering of labels incorrect")
            
            return
        
        self.selected_labels = selected_labels
        
        self.num_classes = len(selected_labels)
        
        initial_subject_list = find_subjects_parallel(data_path)
        
        final_subject_list = []
            
        # Include only the subjects that have a label of interest
        # Separate these into individual lists
        label_subgroups = {label: [] for label in selected_labels}
            
        for subject in initial_subject_list:
            
            label = subject.subject_metadata['Group'].iloc[0]
            
            if label in selected_labels:
                
                label_subgroups[label].append(subject)
        
        # Append the subject list with each label's subgroup
        for label in selected_labels:
            
            final_subject_list.extend(label_subgroups[label])
            
        # Shuffle in case of ordering in directory
        random.shuffle(final_subject_list)
        
        self.subject_list = final_subject_list

    def __len__(self):
        
        return len(self.subject_list)

    def __getitem__(self, index):
        
        subject = self.subject_list[index]
        
        
        """IMAGES"""
        
        # Aligned brain
        brain = self.load_mri_to_tensor(subject.brain_aligned)
        
        # Aligned cropped brain
        brain_cropped = self.load_mri_to_tensor(subject.brain_aligned_cropped)
        
        # NB these are all cropped
        hcampus_vox = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_cropped.nii'))
    
        hcampus_pointcloud = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_cropped_mesh_downsampledcloud.npy')), dtype=torch.float32)
        
        # Won't always have aligned versions
        try:
                               
            hcampus_vox_aligned = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_aligned_cropped.nii'))                        
                    
            hcampus_pointcloud_aligned = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_aligned_cropped_mesh_downsampledcloud.npy')), dtype=torch.float32)
            
        except:
            
            hcampus_vox_aligned, hcampus_pointcloud_aligned = None, None
        
        
        """REGION VOLUME STATS"""
        
        # Get volume column from df
        volume_col = subject.aseg_stats['Volume_mm3']
        
        # Normalise volumes, scale factor to avoid underflow
        volume_col_normalised = volume_col / volume_col.sum() * 1000
        
        
        """SUBJECT INFO - NB NOT COMPLETE WITH SCORES"""
    
        # Info from subject dataframe
        subject_metadata = subject.subject_metadata
        
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
        research_group = mapping.get(subject_metadata['Group'].iloc[0], -1)

        # Return dict with data, this is filtered by collate_fn
        return {
            'brain': brain,
            'brain_cropped': brain_cropped,
            'hcampus_vox': hcampus_vox,
            'hcampus_vox_aligned': hcampus_vox_aligned,
            'hcampus_pointcloud': hcampus_pointcloud,
            'hcampus_pointcloud_aligned': hcampus_pointcloud_aligned,
            'volumes': np.array(volume_col_normalised),
            'research_group': research_group,
        }
        
    def load_mri_to_tensor(self, path):
        
        if path is None or not os.path.isfile(path):
            return torch.empty(0)  # Return empty tensor if the file doesn't exist
        
        # Example of using nibabel to load .mgz files (you can modify as needed)
        image = nibabel.load(path)

        image_data = image.get_fdata()
        
        # Convert to PyTorch tensor
        tensor_data = torch.tensor(image_data, dtype=torch.float32)
        
        return tensor_data

def collate_fn(keys_of_interest):

    def collate_fn_inner(batch):
        
        batch_data = {}

        for key in keys_of_interest:

            if all(key in item for item in batch):

                if isinstance(batch[0][key], torch.Tensor):
                    
                    batch_data[key] = torch.stack([item[key] for item in batch if key in item])
                    
                else:

                    batch_data[key] = torch.tensor([item[key] for item in batch if key in item], dtype=torch.long)
                    
            else:
                
                print(f"Key error")
        
        return batch_data

    return collate_fn_inner