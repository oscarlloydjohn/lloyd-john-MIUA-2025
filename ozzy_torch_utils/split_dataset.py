# Torch
from torch.utils.data import random_split, Subset
import random
import numpy as np

# When prevent_id_leakage is True, no subject id can be split across test and train
# Note that this means that the train/test split may not be exact because the split is
# done on ids which do not always have the same images
def split_dataset(dataset, test_size=0.2, prevent_id_leakage=True):
    
    if prevent_id_leakage:
        
        # Create mapping of ids to indices in the subject_list
        grouped_subjects = {}
            
        for index in range(len(dataset)):
            
            subject = dataset.subject_list[index]
            
            id = subject.subject_metadata['Subject'].iloc[0]
            
            if id not in grouped_subjects:
                
                grouped_subjects[id] = []
                
            grouped_subjects[id].append(index)
        
        # Get the list of ids and shuffle
        ids = list(grouped_subjects.keys())
        
        random.shuffle(ids)
        
        # Split ids
        test_size = int(test_size * len(ids))
                
        test_ids = set(ids[:test_size])
        
        train_ids = set(ids[test_size:])
        
        train_indices, test_indices = [], []
        
        # Convert ids to indices
        for id, indices in grouped_subjects.items():
            
            if id in train_ids:
                
                train_indices.extend(indices)
                
            elif id in test_ids:
                
                test_indices.extend(indices)
        
        # Split dataset using indices
        train_dataset = Subset(dataset, train_indices)
        
        test_dataset = Subset(dataset, test_indices)
        
    else:
    
        dataset_size = len(dataset)
        
        test_size = int(test_size * dataset_size)
        
        train_size = dataset_size - test_size
        
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        
    
    return train_dataset, test_dataset