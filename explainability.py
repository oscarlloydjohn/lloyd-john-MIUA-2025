# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *

import pickle
from captum.attr import *
import random
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

np.set_printoptions(threshold=np.inf)

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.subject_dataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *
from explain_pointnet import *

# Load dataset
data_path = "/uolstore/home/users/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/scratch-disk/full-datasets/hcampus-1.5T-cohort"

subject_list = find_subjects_parallel(data_path)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

subject = sample(subject_list, 1)[0]

# Load model
pickle_pathname = "/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/runs/run_21-03-2025_15-28-42/run_21-03-2025_15-28-42_params.pkl"

with open(pickle_pathname, 'rb') as file:
    
    model_parameters = pickle.load(file)
    
model = model_parameters.model

# Load cloud and explain
cloud = np.load(os.path.join(subject.path, "Left-Hippocampus_aligned_cropped_mesh_downsampledcloud.npy"))

attributions = pointnet_ig(model, cloud, device)

pred_research_group, _, _ = get_prediction(model,cloud, device)

vis_attributions(attributions, subject, cloud, pred_research_group)