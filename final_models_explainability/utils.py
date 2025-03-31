import torch
import random
import numpy as np

def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility in torch, numpy and random. Used for testing

    :param seed: The seed
    :type seed: int
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False