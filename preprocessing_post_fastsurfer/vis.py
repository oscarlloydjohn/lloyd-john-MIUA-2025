"""
Visualisation
===========

This module provides functions for visualising neuroimaging data within python. The functions are designed to be used with the Subject class. Has options for displaying 2D and 3D images, as well as point clouds and meshes.

:author: Oscar Lloyd-John
"""

import nibabel
import nibabel.affines
from PIL import Image
import numpy as np
from IPython.display import display
import pyvista as pv
import matplotlib.pyplot as plt
import os

def display_image(image, clip: bool = True) -> None:

    """
    Display a 2D slice of a 3D image nibabel image. Simply is a wrapper for display_array.

    :param image: The image to be displayed
    :type image: Nibabel image
    :return: None

    """

    # Get image data array from image object
    image_array = np.asarray(image.dataobj)
    
    display_array(image_array, clip=clip)
    
    return 

def display_array(array: np.ndarray, clip: bool = True) -> None:

    """
    
    Display a 2D slice of a 3D array using the IPython display function. The slice is taken in the middle of the array, and is normalised such that the maximum pixel value is 255.

    :param array: The array to be displayed
    :type array: np.ndarray
    :return: None

    """
    
    # Get middle slice
    slice = array[array.shape[0] // 2, :, :]

    if clip:

        # Scale the image such that the maximum pixel value is 255
        display(Image.fromarray(((slice / np.max(slice)) * 255).astype(np.uint8)))

    else:

        img = Image.fromarray(slice.astype(np.uint8))

        display(img)
    
    return 

def display_image_3d(image, downsample_factor: int, mode: str = 'interactive'):

    """
    
    Wrapper for display_array_3d, converts a nibabel image to a numpy array and displays it in 3D.

    :param image: The image to be displayed
    :type image: Nibabel image
    :param downsample_factor: The factor by which to downsample the image
    :type downsample_factor: int
    :param mode: The mode of display, either interactive or preview
    :type mode: str
    :return: None

    """
    
    # Get image data array from image object
    image_array = np.asarray(image.dataobj)
    
    print("converted image to array")
    
    display_array_3d(image_array, downsample_factor, mode)
    
    return


def display_array_3d(array: np.ndarray, downsample_factor: int, mode: str ='interactive') -> None:

    """
    Displays a 3d voxel array using pyvista. The array is downsampled by the downsample factor, and the mode of display can be either interactive or preview (not implemented yet). Interactive allows the user to manipulate the image using the pyvista GUI.

    :param array: The array to be displayed
    :type array: np.ndarray
    :param downsample_factor: The factor by which to downsample the image
    :type downsample_factor: int
    :param mode: The mode of display, either interactive or preview
    :type mode: str
    :return: None
    
    """
    
    # Preview mode is not implemented yet
    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)'''
        
    elif mode == 'interactive':
        
        pass
    
    grid = pv.ImageData()
        
    array = array[::downsample_factor, ::downsample_factor, ::downsample_factor]
    
    grid.dimensions = np.array(array.shape) + 1
    
    grid.cell_data["values"] = array.flatten(order="F")
    
    grid_thresh = grid.threshold(1)
    
    grid_thresh.plot()
    
    return 

def display_cloud(cloud: np.ndarray, mode: str = 'interactive') -> None:

    """
    
    Displays a point cloud of vertices in the form of a numpy array of dimensions n x 3. The mode of display can be either interactive or preview (not implemented yet). Interactive allows the user to manipulate the image using the pyvista GUI. Mode mpl displays the points in a small matplotlib window for quick previewing.

    :param cloud: The point cloud to be displayed
    :type cloud: np.ndarray
    :param mode: The mode of display, either interactive, preview or mpl
    :type mode: str
    :return: None

    """

    mesh = pv.PolyData(cloud)
    
    if mode == 'mpl':

        fig = plt.figure()
        
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c='b', s=1)  # Small point size
        
        plt.show()
        
        return

    # Preview mode is not implemented yet
    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)'''
        
    elif mode == 'interactive':
        
        pass
        
    plotter = pv.Plotter()
    
    plotter.add_mesh(mesh, color='blue', point_size=5, render_points_as_spheres=True)
    
    plotter.set_background("white")
    
    plotter.show()
    
    return

def display_mesh(mesh_dict: dict, mode: str = 'interactive') -> None:

    """
    
    Displays a mesh contained in a dictionary, which will have keys 'verts' and 'faces'. See mesh_creation for more information about the expected format, which will usually be an dict loaded from a .npz file. The mode of display can be either interactive or preview (not implemented yet). Interactive allows the user to manipulate the image using the pyvista GUI.

    :param mesh_dict: The mesh to be displayed
    :type mesh_dict: dict
    :param mode: The mode of display, either interactive or preview
    :type mode: str
    :return: None

    """
    
    # Preview mode is not implemented yet
    if mode == 'preview':
        
        '''plotter = pv.ImageData(window_size=[100, 100], notebook=False)'''
        
    elif mode == 'interactive':
        
        pass
        
    plotter = pv.Plotter()
        
    faces = mesh_dict['faces']
    
    # Convert to correct shape for pyvista
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    
    mesh = pv.PolyData(mesh_dict['verts'], faces_pv)
    
    plotter.add_mesh(mesh, color="lightblue", show_edges=True)
    
    plotter.set_background("white")

    plotter.show()
        
    return
