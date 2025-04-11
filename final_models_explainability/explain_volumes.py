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

    image_array = nibabel.load(brain).dataobj
    aparc_array = nibabel.load(aparc).dataobj

    # Regions and names by absolute shap value
    sorted_indices = np.argsort(np.abs(shap_values.values))[::-1][:5]
    top_features = [shap_values.feature_names[i] for i in sorted_indices]

    plotter = BackgroundPlotter()
    plotter.add_volume(np.asarray(image_array), cmap="bone", opacity="sigmoid_8", show_scalar_bar=False)

    for rank, volume_idx in enumerate(sorted_indices):

        # Segmentation id is different to region index, SegId corresponds to values in the mask
        seg_id = subject.aseg_stats.loc[volume_idx, 'SegId']
        region_name = top_features[rank]

        # Extract region
        mask = np.where(aparc_array == seg_id, 1, 0)
        extracted_region = np.asarray(image_array * mask)

        # Red if positive shap value, blue if negative
        cmap = ListedColormap(['blue']) if shap_values.values[volume_idx] < 0 else ListedColormap(['red'])
        plotter.add_volume(extracted_region, cmap=cmap, opacity="foreground", show_scalar_bar=False, shade=True)

        # Add region label to center of mass of region
        coords = np.argwhere(mask)
        if coords.size > 0:
            center = coords.mean(axis=0)
            plotter.add_point_labels(
                [center],
                [region_name],
                text_color='white',
                point_color='white',
                font_size=12,
                point_size=10,
                shape_opacity=0,
                render_points_as_spheres=True
            )

    plotter.set_background("black")

    plotter.add_text(
    "Red: positive attribution for MCI\nBlue: negative attribution for MCI",
    position='upper_left',
    font_size=12,
    color='white',
    shadow=False
    )

    return plotter
