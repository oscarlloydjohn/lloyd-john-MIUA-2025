
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

# Use pointnet model to run a prediction on a numpy input, and return a string of the research group along with the output as a numpy array
def get_pointnet_prediction(input, device):

    model = pointnet2_cls_ssg.get_model(2, normal_channel=False)

    model.load_state_dict(torch.load("pointnet.pth", weights_only=True))

    if isinstance(input, torch.Tensor):

        input = input.type(torch.float32).to(device)

    elif isinstance(input, np.ndarray):

        input = torch.from_numpy(input).type(torch.float32).to(device)

    else:

        print("Please input numpy array or torch tensor")

    # Add batch dim and transpose to 3 x n for pointnet
    input = input.unsqueeze(0).transpose(2,1)

    model.to(device)

    mapping = {
        0: 'CN',
        1: 'MCI',
    }

    # Remove batch dim
    output = model(input).squeeze(0)

    pred_class = int(torch.argmax(torch.nn.functional.softmax(output, dim=0)).cpu().numpy())

    pred_research_group = mapping.get(pred_class, -1)

    output = output.detach().cpu().numpy()

    return pred_research_group, pred_class, output

def get_volumes_prediction(input):

    with open("volumes_gbdt.pkl", 'rb') as file:
    
        model = pickle.load(file)

    output = model.predict(input)

    return output

def get_scores_prediction(input):

    return

def get_combined_prediction(subject_data):

    _, _, pointnet_output = get_pointnet_prediction(subject_data['cloud'], 'cpu')

    volumes_output = get_volumes_prediction(subject_data['volumes'])

    if subject_data['scores'] is not None:

        scores_output = get_scores_prediction()

        return (pointnet_output[1] + volumes_output + scores_output)/3
    
    else:

        return (pointnet_output[1] + volumes_output)/2