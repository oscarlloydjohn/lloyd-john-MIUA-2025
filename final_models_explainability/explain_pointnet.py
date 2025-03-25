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

def explain_pointnet(model, input):

    # Wrap model as pointnet_cls outputs a tuple for some reason
    wrapped_model = lambda x: model(x)[0]

    ig = IntegratedGradients(wrapped_model)

    # Unsqueeze to add empty batch dimension then transpose  to 3 x n as expected by pointnet
    input = input.unsqueeze(0).transpose(2, 1)

    # Baseline is zeros, however could also be some kind of noise cloud
    baseline = torch.zeros_like(input)

    attributions = ig.attribute(input, baseline, target=1, internal_batch_size=1)
    
    # Transpose back to n x 3 and remove batch dim
    attributions = attributions.transpose(1, 2).squeeze(0)
    
    # Move to CPU for processing
    attributions = attributions.cpu().numpy()
    
    return attributions

def vis_attributions(attributions: np.ndarray, cloud: np.ndarray, power: float = 0.25) -> None:
    
    # Sum x, y and z values for an overall attribution for that point
    xyz_sum = np.sum(attributions, axis=1)

    xyz_sum = np.sign(xyz_sum) * np.power(np.abs(xyz_sum), power)


    # Normalise such that 0 attribution maps to 0.5 and the relative sizes of positive and negative attributions is preserved
    def norm(data):

        min = np.min(data)
        max = np.max(data)

        max_abs_val = np.max((np.abs(min), np.abs(max)))

        return np.array([0.5 + (value / (2 * max_abs_val)) for value in data])


    norm_xyz_sum = norm(xyz_sum)

    # Have to use custom cmap to force the 0 attributions to be white
    colours = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colours)

    pv_cloud = pv.PolyData(cloud)

    plotter = pv.Plotter()

    plotter.add_points(pv_cloud, scalars=norm_xyz_sum, cmap=custom_cmap, clim= [0,1])

    plotter.set_background("black")
    
    return plotter