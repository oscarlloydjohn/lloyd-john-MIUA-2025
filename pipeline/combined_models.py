
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

# Black box wrapper for pointnet, allowing it to take a single input with no batch dimension
class PointNetWrapper(torch.nn.Module):

    def __init__(self, model):
        
        super(PointNetWrapper, self).__init__()

        self.model = model

    def forward(self, x):
        
        # Add batch dim and transpose to 3 x n for pointnet
        x = x.unsqueeze(0).transpose(2, 1)

        # Predict, taking only output and not l3_layer from pointnet
        x = self.model(x)[0]

        return x

# Use pointnet model to run a prediction on a numpy input, and return a string of the research group along with the output as a numpy array
def get_prediction(model, input, device):

    input = torch.from_numpy(input).type(torch.float32).to(device)

    model = PointNetWrapper(model)

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

def get_combined_prediction(subject_data):

    # Load in the model 

    # Run the prediction on the point cloud
    model = pointnet2_cls_ssg.get_model(2, normal_channel=False)

    model.load_state_dict(torch.load("/uolstore/home/student_lnxhome01/sc22olj/Compsci/year3/individual-project-COMP3931/individual-project-sc22olj/pipeline/current_model.pth", weights_only=True))

    # Load in the subject's cloud
    cloud = np.load(os.path.join(subject.path, 'Left-Hippocampus_cropped_mesh_downsampledcloud.npy'))

    # Run the prediction
    _, _, pointnet_output = get_prediction(model, cloud, 'cpu')

    # Predict using the parcellation volumes
    volumes_output = 


    # Predict using the scores
    scores_output = 


    # Average the predictions