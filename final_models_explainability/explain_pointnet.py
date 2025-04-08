"""
Explain pointnet
===========

This module provides a functions for explaining a pointnet classification and visualising the attributions

:author: Oscar Lloyd-John
"""

import torch
from captum.attr import *
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pyvistaqt import BackgroundPlotter

def explain_pointnet(model: torch.nn.Module, input: torch.Tensor) -> np.ndarray:
    """
    Explain an individual classification on a sample by a pointnet model (local explanation) by attributing the output to the input features using integrated gradients

    Transposes the input before attributing as pointnet expects input in the form n x 3. Note that transposing after attribution does not give proper explanations as the attributions will be for dimensions rather than points.

    Note that attributions are always relative to the positive class
    
    :param model: The pointnet model to explain 
    :type model: torch.nn.Module
    :param input: The input to the model
    :type input: torch.Tensor
    :return: The attributions of the input features, the same shape as the input
    :rtype: np.ndarray

    """

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

def normalise_attributions(attributions: np.ndarray, power: float = 0.25) -> np.ndarray:

    """

    Take attributions of shape n x 3 and return a list of attributions for each point. The attributions are summed for each point and then normalised to a value between 0 and 1. 
    
    :param attributions: The pointnet attributions to visualise
    :type attributions: np.ndarray
    :param power: The power to raise the sum of the attributions to before normalising, this can be used to increase the contrast of the attributions, defaults to 0.25
    :type power: float, optional
    :return: A list of point attributions
    :rtype: np.ndarray
    """

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

    return norm_xyz_sum

def vis_attributions(cloud: np.ndarray, norm_xyz_sum: np.ndarray) -> BackgroundPlotter:

    """

    Visualise the normalised attributions returned by normalise_attributions. These values are then visualised as a point cloud with a custom colour map, where 0 attributions are white, positive attributions are red and negative attributions are blue.

    :param cloud: The point cloud to visualise the attributions on, make sure this is the same point cloud that was used to generate the attributions
    :type cloud: np.ndarray
    :param norm_xyz_sum: The normalised attributions
    :type attributions: np.ndarray
    :return: The plotter object used to visualise the attributions
    :rtype: BackgroundPlotter
    """

    # Have to use custom cmap to force the 0 attributions to be white
    colours = [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colours)

    pv_cloud = pv.PolyData(cloud)

    plotter = BackgroundPlotter()

    plotter.add_points(pv_cloud, scalars=norm_xyz_sum, cmap=custom_cmap, clim= [0,1])

    plotter.set_background("black")
    
    return plotter