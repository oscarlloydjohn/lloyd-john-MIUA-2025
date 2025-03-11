# Torch
from torch.utils.data import DataLoader

# Custom modules
from .split_dataset import *
from .SubjectDataset import *

def init_dataloaders(model_parameters, verify_data=False):
    
    for attr, value in vars(model_parameters).items():
        
        if value is None:
            
            print(f"Error: model parameter {attr} incomplete")
            
            return
    
    dataset = SubjectDataset(model_parameters.data_path, model_parameters.selected_labels, downsample_majority=model_parameters.downsample_majority, single_img_per_subject=model_parameters.single_img_per_subject)
    
    # Check the length of the dataset, and the unique labels and IDs
    if verify_data:
        
        print(f"Dataset size: {len(dataset)}\n")

        labels = [dataset[index]['research_group'] for index in range(len(dataset.subject_list))]

        ids = [dataset.subject_list[index].subject_metadata['Subject'] for index in range(len(dataset.subject_list))]

        print(f"Unique labels: {np.unique(labels, return_counts=True)}\n")

        print(f"Unique ids: {np.unique(ids, return_counts=True)}\n")
    
    # Init train and test dataloaders
    train_data, test_data = split_dataset(dataset, test_size=model_parameters.test_size, prevent_id_leakage=model_parameters.prevent_id_leakage)

    # drop_last to prevent last batch tensors from messing up the metrics
    train_dataloader = DataLoader(train_data, batch_size=model_parameters.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn([model_parameters.data_string, model_parameters.labels_string]))

    test_dataloader = DataLoader(test_data, batch_size=model_parameters.batch_size, shuffle=False, drop_last=True, collate_fn=collate_fn([model_parameters.data_string, model_parameters.labels_string]))
    
    # Check the intersection of ids between train and test sets
    if verify_data:
        
        train_ids = [dataset.subject_list[index].subject_metadata['Subject'].iloc[0] for index in train_data.indices]

        test_ids = [dataset.subject_list[index].subject_metadata['Subject'].iloc[0] for index in test_data.indices]

        print(f"Id intersection between train and test: {np.intersect1d(np.unique(train_ids), np.unique(test_ids))}\n")
        
    # Check that no test batch contains only one class as this messes up metrics NB is this a good idea?
    for batch_idx, dict in enumerate(test_dataloader):
        
        if len(np.unique(dict[model_parameters.labels_string])) == 1:
            
            print(f"Labels tensor of batch index {batch_idx} had only one class, retrying\n")     
                  
            train_dataloader, test_dataloader = init_dataloaders(model_parameters, False)
            
            break
        
    return train_dataloader, test_dataloader