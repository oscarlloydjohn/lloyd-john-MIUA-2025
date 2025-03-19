# Torch
import torch
import torch.optim as optim
from torcheval.metrics import *

# Benny pointnet
from pointnet2_benny import pointnet2_cls_msg
import cnn3d_xmuyzz.ResNetV2

import dill as pickle
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

def pointnet_ig(model, cloud, device):
    
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


'''def vis_attributions(attributions, subject, cloud, pred_research_group):
    
    # Sum x, y and z values for an overall attribution for that point
    xyz_sum = np.sum(attributions, axis=1)

    xyz_sum = np.sign(xyz_sum) * np.power(np.abs(xyz_sum), 0.1)

    # Normalise into range -1, 1 such that positive and negative attributions are preserved
    norm = Normalize(vmin = -np.max(np.abs(xyz_sum)), vmax = np.max(np.abs(xyz_sum)))

    norm_attributions = norm(xyz_sum)

    # Cmap for pyvista
    cmap = plt.get_cmap('seismic')
    colours = cmap(norm_attributions)

    pv_cloud = pv.PolyData(cloud)

    plotter = pv.Plotter()

    plotter.add_points(pv_cloud, scalars=colours, rgb=True)

    plotter.set_background("black")

    # THIS IS NOT FOR USE, IT IS RUNNING PREDICTIONS ON TRAINING DATA!!
    plotter.add_text("This is just a test running on training data!", color='white')
    plotter.add_text(f"True class: {str(subject.subject_metadata['Research Group'].iloc[0])} \n Predicted class: {pred_research_group} ", color='white', position='upper_right')

    plotter.show()
    
    return'''

def vis_attributions(attributions, subject, cloud, pred_research_group):
    
    # Sum x, y and z values for an overall attribution for that point
    xyz_sum = np.sum(attributions, axis=1)

    def power_transform(data, power):

        return np.sign(data) * np.power(np.abs(data), power)

    '''plt.plot(xyz_sum)

    plt.show()
    '''

    xyz_sum = power_transform(xyz_sum, 0.25)

    '''plt.plot(xyz_sum)

    plt.show()
    '''

    # Normalise such that 0 attribution maps to 0.5 and the relative sizes of positive and negative attributions is preserved
    def norm(data):

        min = np.min(data)
        max = np.max(data)

        max_abs_val = np.max((np.abs(min), np.abs(max)))

        return np.array([0.5 + (value / (2 * max_abs_val)) for value in data])


    norm_xyz_sum = norm(xyz_sum)

    '''plt.plot(norm_xyz_sum)

    plt.show()
    '''

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