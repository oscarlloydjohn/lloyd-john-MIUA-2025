# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *

# Benny pointnet
from pointnet2_benny import pointnet2_cls_msg
import cnn3d_xmuyzz.ResNetV2

import dill as pickle
import shap
from captum.attr import *
import pyvista as pv
from matplotlib.colors import Normalize
from random import sample
from matplotlib.colors import LinearSegmentedColormap

# Custom modules
from preprocessing_post_fastsurfer.subject import *
from preprocessing_post_fastsurfer.vis import *
from ozzy_torch_utils.split_dataset import *
from ozzy_torch_utils.subject_dataset import *
from ozzy_torch_utils.plot import *
from ozzy_torch_utils.train_nn import *
from ozzy_torch_utils.model_parameters import *
from ozzy_torch_utils.init_dataloaders import *

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

# Use a model to run a prediction, and return a string of the research group along with the tensor
def get_prediction(model, input):

    mapping = {
        0: 'CN',
        1: 'MCI',
    }

    output = model(input)

    pred_class = int(torch.argmax(torch.nn.functional.softmax(output, dim=1)).cpu().numpy())

    pred_research_group = mapping.get(pred_class, -1)

    return pred_research_group, pred_class, output

# Takes a model and a numpy array input and returns a numpy array of attributions
def pointnet_ig(model, cloud, device):

    model.to(device)
    
    model.eval()

    with torch.no_grad():

        wrapped_model = PointNetWrapper(model)

        wrapped_model.to(device)

        ig = IntegratedGradients(wrapped_model)

        # NN expects float32 tensor on device
        input = torch.from_numpy(cloud).type(torch.float32).to(device)

        pred_research_group, pred_class, _ = get_prediction(wrapped_model, input)

        # Baseline is zeros, however could also be some kind of noise cloud
        baseline = torch.zeros_like(input)

        # NB do we always want target to be 1 or the predicted class?
        attributions = ig.attribute(input, baseline, target=1)

        # Move to CPU for processing
        attributions = attributions.cpu().numpy()
    
    return attributions, pred_research_group

## THIS METHOD IS BETTER, BECAUSE WE WANT TO PASS THE MATRIX IN WHERE POINTS ARE FEATURES OTHERWISE IG WILL BE CALCULATING THE IMPORTANCE OF X,Y,Z
def pointnet_ig_old(model, cloud, device):

    model.to(device)
    
    model.eval()

    with torch.no_grad():

        # Wrap model as pointnet_cls outputs a tuple for some reason
        wrapped_model = lambda x: model(x)[0]

        ig = IntegratedGradients(wrapped_model)

        input = torch.from_numpy(cloud)

        # NN expects float32 on cuda
        input = input.type(torch.float32).to(device)

        # Unsqueeze to add empty batch dimension then transpose  to 3 x n as expected by pointnet
        input = input.unsqueeze(0).transpose(2, 1)
        
        output = wrapped_model(input)
        
        prediction = torch.argmax(torch.nn.functional.softmax(output, dim=1)).cpu().numpy()
        
        mapping = {
            0: 'CN',
            1: 'MCI',
        }
            
        # Get the value of the mapping, -1 if not found
        pred_research_group = mapping.get(int(prediction), -1)

        # Baseline is zeros, however could also be some kind of noise cloud
        baseline = torch.zeros_like(input)

        attributions = ig.attribute(input, baseline, target=1, internal_batch_size=1)
        
        # Transpose back to n x 3 and remove batch dim
        attributions = attributions.transpose(1, 2).squeeze(0)
        
        # Move to CPU for processing
        attributions = attributions.cpu().numpy()
    
    return attributions, pred_research_group


# Takes a model and a numpy array input and returns a numpy array of attributions
def pointnet_saliency(model, cloud, device):

    model.to(device)
    
    model.eval()

    wrapped_model = PointNetWrapper(model)

    wrapped_model.to(device)

    ig = Saliency(wrapped_model)

    # NN expects float32 tensor on device
    input = torch.from_numpy(cloud).type(torch.float32).to(device)
    
    pred_research_group, pred_class, _ = get_prediction(wrapped_model, input)

    # NB do we always want target to be 1 or the predicted class?
    attributions = ig.attribute(input, target=1, abs=False)
    
    # Move to CPU for processing
    attributions = attributions.cpu().numpy()
    
    return attributions, pred_research_group

def vis_attributions(attributions: np.ndarray, subject: Subject, cloud: np.ndarray, pred_research_group: str, plot_attributions: bool = False, power: float = 0.25) -> None:
    
    # Sum x, y and z values for an overall attribution for that point
    xyz_sum = np.sum(attributions, axis=1)

    if plot_attributions:

        plt.plot(xyz_sum)

        plt.show()

    xyz_sum = np.sign(xyz_sum) * np.power(np.abs(xyz_sum), power)

    if plot_attributions:
        
        plt.plot(xyz_sum)

        plt.show()

    # Normalise such that 0 attribution maps to 0.5 and the relative sizes of positive and negative attributions is preserved
    def norm(data):

        min = np.min(data)
        max = np.max(data)

        max_abs_val = np.max((np.abs(min), np.abs(max)))

        return np.array([0.5 + (value / (2 * max_abs_val)) for value in data])


    norm_xyz_sum = norm(xyz_sum)

    if plot_attributions:
        
        plt.plot(xyz_sum)

        plt.show()

    # Have to use custom cmap to force the 0 attributions to be white
    colours = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colours)

    pv_cloud = pv.PolyData(cloud)

    plotter = pv.Plotter()

    plotter.add_points(pv_cloud, scalars=norm_xyz_sum, cmap=custom_cmap, clim= [0,1])

    plotter.set_background("black")

    # THIS IS NOT FOR USE, IT IS RUNNING PREDICTIONS ON TRAINING DATA!!
    plotter.add_text("This is just a test running on training data!", color='white')
    plotter.add_text(f"True class: {str(subject.subject_metadata['Research Group'].iloc[0])} \n Predicted class: {pred_research_group} ", color='white', position='upper_right')

    plotter.show()
    
    return