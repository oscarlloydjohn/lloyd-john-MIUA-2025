# Torch
from torch.utils.data import DataLoader

# Custom modules
from .split_dataset import *
from .subject_dataset import *

def init_dataloaders(model_parameters, num_workers = 4, load_in_memory = False, verify_data=False):
    
    for attr, value in vars(model_parameters).items():
        
        # Criterion is the only parameter that needs to be set after dataloader init
        if attr == 'criterion':
            
            continue
        
        if value is None:
            
            print(f"Error: model parameter {attr} incomplete")
            
            return
    
    dataset = SubjectDataset(model_parameters.data_path, model_parameters.selected_labels, load_in_memory = load_in_memory)
    
    # Check the length of the dataset, and the unique labels and IDs
    if verify_data:
        
        print(f"Dataset size: {len(dataset)}\n")

        labels = [dataset[index]['research_group'] for index in range(len(dataset.subject_list))]

        ids = [dataset.subject_list[index].subject_metadata['Subject'] for index in range(len(dataset.subject_list))]

        print(f"Unique labels: {np.unique(labels, return_counts=True)}\n")

        print(f"Unique ids: {np.unique(ids, return_counts=True)}\n")
    
    # Init train and test dataloaders
    train_data, test_data = split_dataset(dataset, test_size=model_parameters.test_size)

    # drop_last to prevent last batch tensors from messing up the metrics
    train_dataloader = DataLoader(train_data, batch_size=model_parameters.batch_size, shuffle=True, num_workers = num_workers, drop_last=model_parameters.drop_last, collate_fn=collate_fn([model_parameters.data_string, model_parameters.labels_string]))

    test_dataloader = DataLoader(test_data, batch_size=model_parameters.batch_size, shuffle=False, num_workers = num_workers, drop_last=model_parameters.drop_last, collate_fn=collate_fn([model_parameters.data_string, model_parameters.labels_string]))
    
    # Check the intersection of ids between train and test sets
    if verify_data:
        
        train_ids = [dataset.subject_list[index].subject_metadata['Subject'].iloc[0] for index in train_data.indices]

        test_ids = [dataset.subject_list[index].subject_metadata['Subject'].iloc[0] for index in test_data.indices]

        print(f"Id intersection between train and test: {np.intersect1d(np.unique(train_ids), np.unique(test_ids))}\n")
        
    return train_dataloader, test_dataloader

def get_weights(train_dataloader):
    
    labels = torch.empty(0, dtype=torch.int32)
    
    for batch_idx, dict in enumerate(train_dataloader):
        
        labels = torch.cat((labels, dict['research_group']))
        
    counts = torch.bincount(labels)
    
    weights = len(labels) / counts.float()
        
    return weights
        
    