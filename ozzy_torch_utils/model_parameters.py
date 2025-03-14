import os
from typing import Callable

class ModelParameters:
    
    data_path: os.PathLike[str]
    selected_labels: list
    data_string: str
    labels_string: str
    batch_size: int
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
        self.test_size = None
        self.num_epochs = None
        self.learning_rate = None
        self.model = None
        self.criterion = None
        self.optimiser = None
        self.run_prediction = None