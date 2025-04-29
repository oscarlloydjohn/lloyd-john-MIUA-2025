"""

Model parameters
===========

This is a class used to store all hyperparameters used in the train_nn module. It also stores the model, criterion and optimiser objects, and a function to run predictions on the model.

:author: Oscar Lloyd-John

"""

import os
from typing import Callable

class ModelParameters:

    """
    
    :ivar data_path: The path to the dataset, as used by init_subjects
    :vartype data_path: os.PathLike[str]
    :ivar selected_labels: The labels to be used in the model, a list containing strings CN, MCI, and/or AD (must start with CN as this is assumed to be class 0)
    :vartype selected_labels: list
    :ivar data_string: The dict key we are interested in from the dict returned by SubjectDataset.__getitem__. Could be hippocampus, brain or point cloud for example
    :vartype data_string: str
    :ivar labels_string: A second key we are interested in from the dict returned by SubjectDataset.__getitem__. Will always be 'research_group' for this project.
    :vartype labels_string: str
    :ivar batch_size: The batch size to be used in the DataLoader
    :vartype batch_size: int
    :ivar drop_last: Whether to drop the last batch in the DataLoader if it is smaller than the batch size. Can be useful when calculating metrics although current metrics implementation does not need it 
    :vartype drop_last: bool
    :ivar test_size: The proportion of the dataset to be used for testing
    :vartype test_size: float
    :ivar num_epochs: The number of epochs to train the model for
    :vartype num_epochs: int
    :ivar learning_rate: The learning rate for the optimiser
    :vartype learning_rate: float
    :ivar model: The model object
    :vartype model: object
    :ivar criterion: The loss function object
    :vartype criterion: object
    :ivar optimiser: The optimiser object
    :vartype optimiser: object
    :ivar run_prediction: The function to call the model with the input data, and return the output values and predicted classes. This will depend on the model used and the shape of the input. For example:

    .. code-block:: python

        def run_prediction(inputs, labels):
        
            inputs = inputs.transpose(2, 1)
            
            logit_output, *_ = model_parameters.model(inputs)
            
            loss = model_parameters.criterion(logit_output, labels)
            
            # Apply exponent as the output of the model is log softmax
            pred_probability = torch.exp(logit_output)
                
            # Threshold is variable to give preference to FN or FP
            pred_labels = (pred_probability[:, 1] >= 0.5).int()
            
            return loss, pred_probability, pred_labels

    :vartype run_prediction: Callable

    """
    
    data_path: os.PathLike[str]
    selected_labels: list
    data_string: str
    labels_string: str
    batch_size: int
    drop_last: bool
    test_size: float
    num_epochs: int
    learning_rate: float
    model: object
    criterion: object
    optimiser: object
    run_prediction: Callable
    
    def __init__(self):
        self.data_path = None
        self.selected_labels = None
        self.data_string = None
        self.labels_string = None
        self.batch_size = None
        self.drop_last = None
        self.test_size = None
        self.num_epochs = None
        self.learning_rate = None
        self.model = None
        self.criterion = None
        self.optimiser = None
        self.run_prediction = None
        