"""

Split dataset
===========

:author: Oscar Lloyd-John

"""

from torch.utils.data import random_split, Subset, Dataset

def split_dataset(dataset: Dataset, test_size: float = 0.2) -> tuple[Subset, Subset]:

    """

    Simple function to perform a random split into train and test datasets. This function used to have functionality to prevent ID leakage when there were mutiple images per subject however it is assumed that only one image per subject will be studied in the current project.

    :param dataset: The dataset to be split
    :type dataset: Dataset
    :param test_size: The proportion of the dataset to be used for testing
    :type test_size: float
    :return: The training and testing datasets
    :rtype: tuple[Subset, Subset]

    """

    dataset_size = len(dataset)
    
    test_size = int(test_size * dataset_size)
    
    train_size = dataset_size - test_size
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    return train_dataset, test_dataset