# Torch
from torch.utils.data import DataLoader

# Custom modules
from .split_dataset import *
from .subject_dataset import *

"""

Initialise dataloaders
===========

Initialises train and test datasets for use in train_nn. These are created from the SubjectDataset class.

:author: Oscar Lloyd-John

"""

def init_dataloaders(model_parameters, num_workers = 4, load_in_memory = False, verify_data=False) -> tuple[DataLoader, DataLoader]:
    
    """
    
    Given the parameters from the ModelParameters object, initalises a pair of dataloaders from the SubjectDataset dataset. The function also checks the dataset for any issues such as ID leakage between the train and test sets, and the distribution of labels.

    All model_parameters parameters must be set beforehand, except for criterion which is set after dataloader init as it is dependent on the distribution of labels in the loaders (for the weights).

    Dataloader can be checked for common issues (often causing overfitting) in MRI classification such as subject ID leakage between train and test sets. Distribution of labels is also checked.

    :param model_parameters: The model parameters to be used for training, including the model and criterion objects. See model_parameters.py. 
    :type model_parameters: ModelParameters
    :param num_workers: The number of workers (threads) to be used to create the dataloader
    :type num_workers: int
    :param load_in_memory: Whether to load the entire dataset into memory. This can be useful for small datasets, but not for large datasets. It is slow to load however speeds up training especially if training a large number of models that read data from the network.
    :type load_in_memory: bool
    :param verify_data: Print out the dataset size, unique labels, and unique ids
    :type verify_data: bool
    :return: The training and testing dataloaders
    :rtype: tuple[DataLoader, DataLoader]

    """
    for attr, value in vars(model_parameters).items():
        
        # Criterion is the only parameter that needs to be set after dataloader init
        if attr == 'criterion':
            
            continue
        
        if value is None:
            
            print(f"Error: model parameter {attr} incomplete")
            
            return
    
    if load_in_memory:
        
        dataset = SubjectDataset(model_parameters.data_path, model_parameters.selected_labels, load_in_memory = load_in_memory, data_string=model_parameters.data_string, labels_string=model_parameters.labels_string)
        
    else:
        
        dataset = SubjectDataset(model_parameters.data_path, model_parameters.selected_labels, load_in_memory = load_in_memory)
    
    # Check the length of the dataset, and the unique labels and IDs
    if verify_data:
        
        print(f"Dataset size: {len(dataset)}\n")

        labels = [dataset[index]['research_group'] for index in range(len(dataset.subject_list))]

        ids = [dataset.subject_list[index].subject_metadata['Subject ID'] for index in range(len(dataset.subject_list))]

        print(f"Unique labels: {np.unique(labels, return_counts=True)}\n")

        print(f"Unique ids: {np.unique(ids, return_counts=True)}\n")
    
    # Init train and test dataloaders
    train_data, test_data = split_dataset(dataset, test_size=model_parameters.test_size)

    # drop_last to prevent last batch tensors from messing up the metrics
    train_dataloader = DataLoader(train_data, batch_size=model_parameters.batch_size, shuffle=True, num_workers = num_workers, drop_last=model_parameters.drop_last)

    test_dataloader = DataLoader(test_data, batch_size=model_parameters.batch_size, shuffle=False, num_workers = num_workers, drop_last=model_parameters.drop_last)
    
    # Check the intersection of ids between train and test sets
    if verify_data:
        
        train_ids = [dataset.subject_list[index].subject_metadata['Subject ID'].iloc[0] for index in train_data.indices]

        test_ids = [dataset.subject_list[index].subject_metadata['Subject ID'].iloc[0] for index in test_data.indices]

        print(f"Id intersection between train and test: {np.intersect1d(np.unique(train_ids), np.unique(test_ids))}\n")
        
    return train_dataloader, test_dataloader

def get_weights(train_dataloader):

    """
    Get the class weights from the dataloader to be passed to a criterion object. Used as ADNI has unbalanced datasets.

    :param train_dataloader: The training dataloader
    :type train_dataloader: DataLoader
    :return: A tensor of the class weights
    :rtype: torch.Tensor
    """
    
    labels = torch.empty(0, dtype=torch.int32)
    
    for batch_idx, dict in enumerate(train_dataloader):
        
        labels = torch.cat((labels, dict['research_group']))
        
    counts = torch.bincount(labels)
    
    weights = len(labels) / counts.float()
        
    return weights
        
    