# Torch
from torch.utils.data import random_split, Subset
import random
import numpy as np

def split_dataset(dataset, test_size=0.2):

    dataset_size = len(dataset)
    
    test_size = int(test_size * dataset_size)
    
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset