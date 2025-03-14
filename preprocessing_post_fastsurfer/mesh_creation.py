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

# Convert a volume of voxels to a pointcloud mesh using walking cubes
def volume_to_mesh(subject: object, fname_or_attribute, smooth: bool = False, number_of_iterations: int = 1, lambda_filter: float = 0.5, **kwargs) -> dict:
    
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
    
    min_samples = np.inf
    
    for subject in subject_list:
        
        min_samples = min(min_samples, len(np.load(os.path.join(subject.path, filename), allow_pickle=True)['verts']))
    
    return min_samples

# https://medium.com/towards-data-science/how-to-use-pointnet-for-3d-computer-vision-in-an-industrial-context-3568ba37327e
# Farthest point sampling is suggested
# NB used open3d as it natively supports this type of sampling
def downsample_cloud(subject: Subject, filename, n):
    
    mesh_dict = np.load(os.path.join(subject.path, filename))
    
    cloud = o3d.geometry.PointCloud()
    
    cloud.points = o3d.utility.Vector3dVector(mesh_dict['verts'])

    downsampled_cloud = np.asarray(cloud.farthest_point_down_sample(n).points)
    
    basename = os.path.splitext(os.path.basename(filename))[0]
        
    downsampled_path = os.path.join(subject.path,(basename + '_downsampledcloud.npy'))
    
    np.save(downsampled_path, downsampled_cloud)
    
    return downsampled_cloud

def downsample_cloud_parallel(subject_list, filename, num_samples, display=False):
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        
        futures = []
        
        for subject in subject_list:

            futures.append(executor.submit(downsample_cloud, subject, filename, num_samples))
            
        for future in concurrent.futures.as_completed(futures):
            
            cloud = future.result()
            
            if display:
                
                display_cloud(cloud, mode='mpl')
    
    return
