"""
Explain pointnet
===========

This module provides a functions for explaining a pointnet classification and visualising the attributions

:author: Oscar Lloyd-John
"""

import numpy as np
import pyvista as pv
from random import sample
import shap
from matplotlib.colors import ListedColormap
from pyvistaqt import BackgroundPlotter
import sklearn

# Custom modules
from preprocessing_post_fastsurfer.extraction import *
from preprocessing_post_fastsurfer.subject import *

def explain_volumes(model: sklearn.ensemble.GradientBoostingClassifier, volumes: np.ndarray, struct_names: list) -> shap.Explainer:
    """
    Explain an individual classification on a sample by the parcellation volumes gradient boosted classifier. 

    :param model: The gradient boosted classifier model to explain (parcellation volumes model)
    :type model: sklearn.ensemble.GradientBoostingClassifier
    :param volumes: Single sample of parcellation volumes to explain
    :type volumes: np.ndarray
    :param struct_names: Names of the structures in the volumes
    :type struct_names: list
    :return: The shap values of the volumes
    :rtype: shap.Explainer
    """
    volumes = volumes.squeeze(0)

    explainer = shap.Explainer(model, feature_names=struct_names)

    shap_values = explainer(volumes)

    return shap_values

import numpy as np
import nibabel
import pyvista as pv

def vis_volumes(subject: Subject, shap_values: shap.Explainer) -> BackgroundPlotter:
    """
    Visualise the shap values of the parcellation volumes for a single sample. The top 5 volumes are visualised in the actual subject's brain volume, with the volumes coloured by the shap value.

    :param subject: The subject to visualise the volumes on
    :type subject: Subject
    :param shap_values: The shap values of the volumes
    :type shap_values: shap.Explainer
    :return: The plotter with the volumes visualised
    :rtype: BackgroundPlotter
    """

    brain = subject.brain_aligned
    aparc = subject.aparc_aligned

    # NB check types here!
    image_array = nibabel.load(brain).dataobj
    aparc_array = nibabel.load(aparc).dataobj

    # Get only the values
    shap_values = shap_values.values

    top_volumes = np.argsort(np.abs(shap_values))[::-1][:5]

    plotter = BackgroundPlotter()

    plotter.add_volume(np.asarray(image_array), cmap="bone", opacity="sigmoid_8", show_scalar_bar=False)

    for volume in top_volumes:

        # Convert volume index to segmentation region id
        id = subject.aseg_stats.loc[volume, 'SegId']

        # Create a mask from seg ids rather than volume index
        filtered_array = np.where((np.isin(aparc_array, [id])), 1, 0)

        # Extract region using mask
        extracted_region = np.asarray(image_array * filtered_array)

        if shap_values[volume] < 0:
            cmap = ListedColormap(['blue'])
        else:
            cmap = ListedColormap(['red'])

        plotter.add_volume(extracted_region, cmap=cmap, opacity="foreground", show_scalar_bar=False, shade=True)
        
    plotter.set_background("black")

    return plotter
