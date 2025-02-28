
# Torch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.optim as optim
from torcheval.metrics import *

# Benny pointnet
from pointnet2_benny import pointnet2_cls_msg
from pointnet2_benny import provider

# Other
from tqdm import tqdm
import nibabel
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, accuracy_score, roc_auc_score
from collections import Counter
import random
import pandas as pd

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
        
        subject_list = []
        
        # Include only the subjects that have a label of interest
        # Separate these into individual lists
        label_subgroups = {label: [] for label in selected_labels}
            
        for subject in find_subjects(data_path):
            
            label = subject.subject_metadata['Group'].iloc[0]
            
            if label in selected_labels:
                
                label_subgroups[label].append(subject)
                
        # Size of smallest label subgroup
        min_class_size = min(len(label_subgroups[label]) for label in selected_labels)
            
        # Downsample the list to reduce class imbalance
        if downsample_majority:
            
            for label in selected_labels:
                
                if len(label_subgroups[label]) > min_class_size:
                    
                    label_subgroups[label] = random.sample(label_subgroups[label], min_class_size)
        
        
        # Append the subject list with each label's subgroup
        for label in selected_labels:
            
            subject_list.extend(label_subgroups[label])
            
        # Shuffle in case of ordering in directory
        random.shuffle(subject_list)
        
        self.subject_list = subject_list

    def __len__(self):
        
        return len(self.subject_list)

    def __getitem__(self, index):
        
        subject = self.subject_list[index]
        
        
        """IMAGES"""
        
        # Aligned cropped brain
        brain = self.load_mri_to_tensor(subject.brain_aligned_cropped)
        
        # NB these are all cropped
        hcampus_vox = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_cropped.nii'))
        
        hcampus_vox_aligned = self.load_mri_to_tensor(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_aligned_cropped.nii'))
    
        hcampus_pointcloud = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_cropped_mesh_downsampled.npy')), dtype=torch.float32)
                                          
        hcampus_pointcloud_aligned = torch.tensor(np.load(os.path.join(subject.path, 'Left-Hippocampus_Right-Hippocampus_aligned_cropped_mesh_downsampled.npy')), dtype=torch.float32)
        
        
        """REGION VOLUME STATS"""
        
        # Need to process stats as cannot put dataframe as tensor 
        aseg_stats = subject.aseg_stats
        
        
        """SUBJECT INFO - NB NOT COMPLETE WITH SCORES, NEED TO PARSE XML"""
    
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
            'hcampus_vox': hcampus_vox,
            'hcampus_vox_aligned': hcampus_vox_aligned,
            'hcampus_pointcloud': hcampus_pointcloud,
            'hcampus_pointcloud_aligned': hcampus_pointcloud_aligned,
            #'aseg_stats': aseg_stats, # Cannot convert dataframe to tensor, need to process
            'research_group': research_group,
            'sex': subject_metadata['Sex'].iloc[0],
            'age': subject_metadata['Age'].iloc[0],
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