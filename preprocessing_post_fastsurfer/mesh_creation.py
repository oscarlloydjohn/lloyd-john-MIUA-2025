"""
Mesh creation
===========

This module provides functions for converting a voxel volumes into point clouds and manipulating these clouds. It is intended to be used for single brain regions such as the hippocampus however would theoretically work on the whole brain. It is designed to be used with the Subject class. The number of points can be specified in these functions such that the point clouds can be downsampled to a consistent size for use in neural networks.

:author: Oscar Lloyd-John
"""

import nibabel
import nibabel.affines
import os
import numpy as np
import concurrent.futures
from skimage import measure
import open3d as o3d

# Custom modules
from .vis import *
from .subject import *

def volume_to_mesh(subject: object, fname_or_attribute, smooth: bool = False, number_of_iterations: int = 1, lambda_filter: float = 0.5, **kwargs) -> dict:

    """

    Converts a 3D array of voxels to a point cloud mesh using the marching cubes algorithm. The mesh is saved as a .npz file in the subject directory, which contains all outputs from the skimage marching cubes algorithm.

    The mesh can be smoothed using open3d laplacian filter, which is useful for reducing the blocky appearance from the large voxel size without destroying the shape. In this case, the mesh is saved back in the same format, however the surface values are lost.

    Meshes are saved as the input filename plus '_mesh.npz'. This is a serialised numpy dictionary containing the vertices, faces, normals and values of the mesh.

    :param subject: The subject containing the image to be converted
    :type subject: Subject
    :param fname_or_attribute: The filename or attribute of the image to be converted, accessed using getattr
    :type fname_or_attribute: str
    :param smooth: Whether or not to smooth the mesh using open3d, defaults to False
    :type smooth: bool
    :param number_of_iterations: The number of laplacian filter iterations to smooth the mesh, defaults to 1
    :type number_of_iterations: int
    :param lambda_filter: The lambda filter parameter for the laplacian filter, defaults to 0.5
    :type lambda_filter: float
    :param kwargs: Any keyword arguments to be passed to the marching cubes algorithm
    :return: A dictionary containing the vertices, faces, normals and values of the mesh
    :rtype: dict

    """
    
    # Allows filenames or Subject object attributes for convenience
    if os.path.exists(os.path.join(subject.path, fname_or_attribute)):
        
        image_array = nibabel.load(os.path.join(subject.path, fname_or_attribute)).get_fdata()
        
    else:
        
        image_array = nibabel.load(getattr(subject, fname_or_attribute)).get_fdata()
    
    # Convert the volume to a mesh
    verts, faces, normals, values = measure.marching_cubes(image_array, **kwargs)
    
    # Smooth using open3d as sklearn does not have smoothing
    # Saved back into the same format although values are lost
    if smooth:
        
        mesh = o3d.geometry.TriangleMesh()
        
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
        
        mesh = mesh.filter_smooth_laplacian(number_of_iterations = number_of_iterations, lambda_filter = lambda_filter)
        
        mesh.compute_vertex_normals()
        
        verts = np.asarray(mesh.vertices)
        
        faces = np.asarray(mesh.triangles)
        
        normals = np.asarray(mesh.vertex_normals)
        
        # Values are discarded
        values = None
    
    # Save as dict in npz file for easy loading
    fname = os.path.splitext(os.path.basename(fname_or_attribute))[0]
    
    pathname = os.path.join(subject.path, fname + '_mesh.npz')
    
    np.savez(pathname, verts=verts, faces=faces, normals=normals, values=values)
    
    return {"verts" : verts, "faces" : faces, "normals" : normals, "values" : values}

def volume_to_mesh_parallel(subject_list: list, fname_or_attribute, display: bool = False, smooth: bool = False, number_of_iterations: int = 1, lambda_filter: float = 0.5, **kwargs) -> None:

    """
    
    Converts a list of 3D arrays of voxels to point cloud meshes in parallel using a processpoolexecutor. Simply calls volume_to_mesh on each subject in the list.

    :param subject_list: The list of subjects to have their images converted
    :type subject_list: list
    :param fname_or_attribute: The filename or attribute of the image to be converted, accessed using getattr
    :type fname_or_attribute: str
    :param display: Whether or not to display the mesh upon creation, defaults to False
    :type display: bool
    :param smooth: Whether or not to smooth the mesh using open3d, defaults to False
    :type smooth: bool
    :param number_of_iterations: The number of laplacian filter iterations to smooth the mesh, defaults to 1
    :type number_of_iterations: int
    :param lambda_filter: The lambda filter parameter for the laplacian filter, defaults to 0.5
    :type lambda_filter: float
    :param kwargs: Any keyword arguments to be passed to the marching cubes algorithm
    :return: None

    """
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(volume_to_mesh, subject, fname_or_attribute, smooth, number_of_iterations, lambda_filter, **kwargs))
            
        for future in concurrent.futures.as_completed(futures):
            
            mesh = future.result()
            
            if display:
                
                display_mesh(mesh, mode='preview')
    
    return
    
# Get the minimum number of vertices over all meshes in the list
def get_min_cloud_points(subject_list: list, filename: str) -> int:

    """
    Finds the minimum number of vertices of a given filename in a list of subjects. Useful for ensuring that downsampling will be successful and that all meshes are of the same size (for neural networks).

    :param subject_list: The list of subjects
    :type subject_list: list
    :param filename: The filename to check number of vertices for
    :type filename: str
    :return: The minimum number of vertices in that filename in the list
    :rtype: int

    """
    
    min_samples = np.inf
    
    for subject in subject_list:
        
        min_samples = min(min_samples, len(np.load(os.path.join(subject.path, filename), allow_pickle=True)['verts']))
    
    return min_samples

def downsample_cloud(subject: Subject, filename: str, n: int) -> np.ndarray:

    """
    
    Downsamples a point cloud to n points using open3d farthest point sampling. The downsampled cloud is saved as a .npy file in the subject directory, with the filename appended with '_downsampledcloud.npy'. Uses farthest point sampling rather than random sampling such that shape is preserved.

    See https://medium.com/towards-data-science/how-to-use-pointnet-for-3d-computer-vision-in-an-industrial-context-3568ba37327e for more information. Used open3d as it is one of the only python libraries that supports farthest point sampling, where the number of points can be specified rather than a fraction.

    :param subject: The subject containing the point cloud to be downsampled
    :type subject: Subject
    :param filename: The filename of the point cloud to be downsampled
    :type filename: str
    :param n: The number of points to downsample to
    :type n: int
    :return: The downsampled point cloud
    :rtype: np.ndarray

    """
    
    mesh_dict = np.load(os.path.join(subject.path, filename))
    
    cloud = o3d.geometry.PointCloud()
    
    cloud.points = o3d.utility.Vector3dVector(mesh_dict['verts'])

    downsampled_cloud = np.asarray(cloud.farthest_point_down_sample(n).points)
    
    basename = os.path.splitext(os.path.basename(filename))[0]
        
    downsampled_path = os.path.join(subject.path,(basename + '_downsampledcloud.npy'))
    
    np.save(downsampled_path, downsampled_cloud)
    
    return downsampled_cloud

def downsample_cloud_parallel(subject_list: list[Subject], filename: str, num_samples: int, display: bool = False) -> None:

    """
    
    Downsamples a list of point clouds in parallel using a processpoolexecutor. Simply calls downsample_cloud on each subject in the list.

    :param subject_list: The list of subjects to have their point clouds downsampled
    :type subject_list: list[Subject]
    :param filename: The filename of the point cloud to be downsampled
    :type filename: str
    :param num_samples: The number of points to downsample to
    :type num_samples: int
    :param display: Whether or not to display the downsampled cloud upon creation, defaults to False
    :type display: bool
    :return: None

    """
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(downsample_cloud, subject, filename, num_samples))
            
        for future in concurrent.futures.as_completed(futures):
            
            cloud = future.result()
            
            if display:
                
                display_cloud(cloud, mode='mpl')
    
    return
