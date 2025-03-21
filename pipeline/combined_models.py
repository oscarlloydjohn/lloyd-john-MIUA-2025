
# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *
import torch.nn.functional as F

# Benny pointnet
from pointnet2_benny import pointnet2_cls_ssg
import shutil

# Custom modules
from preprocessing_pre_fastsufer.preprocess import *
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.subject_dataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *

from explain_pointnet import get_prediction

# Load pytorch model


model = pointnet2_cls_ssg.get_model(2, normal_channel=False)

model.load_state_dict(torch.load("/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/current_model.pth", weights_only=True))

get_prediction(model, cloud, 'cpu')